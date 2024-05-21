from concurrent.futures import ThreadPoolExecutor
import modal
from modal import Secret, Stub, gpu, build, enter, method
from typing import List, Tuple, Any
import numpy as np
from .common import cosine_similarity
import os

import modal

stub = Stub("vpr")

inference_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.2.2",
    "pillow==10.3.0",
    "torchvision==0.17.2",
)

with inference_image.imports():
    import torch
    from torchvision.transforms import transforms
    from PIL import Image
    from io import BytesIO


ImgType = bytes | np.ndarray | Image.Image

@stub.cls(gpu=gpu.A10G(), image=inference_image, enable_memory_snapshot=True)
class VPRModel:
    @build()
    def build(self):
        _ = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )

    @enter(snap=True)
    def load(self):
        self.model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )

        self.base_transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                # transforms.Resize((644, 644))
            ]
        )
    
    @enter(snap=False)
    def setup(self):
        self.model = self.model.cuda()

    @method()
    def inference(self, image: ImgType, panos: List[ImgType], n: int = 5, batch_size=50):
        import time
        t_loading = 0
        t_transforms = 0
        t_inference = 0
        t_beg = time.time()
        pano_tensors = []
        print(f"running inference on {len(panos)} panos")
        tns_transform = transforms.ToTensor()

        def load_and_transform(image: ImgType):
            with torch.no_grad():
                if isinstance(image, bytes):
                    image = Image.open(BytesIO(image)).convert("RGB")
                tensor = tns_transform(image)
                return tensor

        for i in range(0, len(panos), batch_size):
            batch = panos[i : i + batch_size]
            with torch.no_grad():
                t_0 = time.time()
                with ThreadPoolExecutor() as executor:
                    loaded_images = torch.stack(list(executor.map(load_and_transform, batch)))
                t_ld = time.time()
                t_loading += t_ld - t_0
                inp = self.base_transform(loaded_images).to("cuda")
                del loaded_images
                t_1 = time.time()
                t_transforms += t_1 - t_ld
                batch_tensor = self.model(inp)
                t_2 = time.time()
                t_inference += t_2 - t_1
                del inp
                pano_tensors.append(batch_tensor)
                print(f"batch {i} done")
                del batch
        pano_tensor = torch.cat(pano_tensors, dim=0)
        with torch.no_grad():
            img_tensor = self.model(
                torch.stack(
                    [self.base_transform(load_and_transform(image))]
                ).to("cuda")
            )

        print(f"transforms: {t_transforms}s, inference: {t_inference}s, loading: {t_loading}s, total: {time.time() - t_beg}s")
        cos_sim = cosine_similarity(img_tensor, pano_tensor)
        return cos_sim.tolist()[0]


@stub.local_entrypoint()
def main():
    import os
    import glob
    from PIL import Image
    src_path = ".images/sv_test"
    src_locs = glob.glob(f"{src_path}/db/*.png")
    im_db = [np.asarray(Image.open(loc)) for loc in src_locs]
    DB_SIZE = 600
    im_db *= (DB_SIZE // len(im_db) + 1)
    im_db = im_db[:DB_SIZE]
    tar = np.asarray(Image.open(f"{src_path}/targ.png"))
    print(f"{len(im_db)} Images Loaded.")
    res = VPRModel().inference.remote(tar, im_db)
    print(res)
