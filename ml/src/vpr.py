import modal
from modal import Secret, Stub, gpu, build, enter, method
from typing import List, Tuple, Any
import asyncio
from .common import cosine_similarity
import os

import modal

stub = Stub("vpr")

inference_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.2.2",
    "pillow==10.3.0",
    "torchvision==0.17.2",
    "pytorch-lightning",
    "pytorch-metric-learning",
    "faiss-gpu",
    "torchmetrics",
    "prettytable",
    "xformers"
)

with inference_image.imports():
    import torch
    from torchvision.transforms import transforms
    from PIL import Image
    from io import BytesIO


@stub.cls(gpu=gpu.A10G(), image=inference_image)
class VPRModel:
    @build()
    def build(self):
        _ = torch.hub.load("serizba/salad", "dinov2_salad")

    @enter()
    def enter(self):
        self.model = torch.hub.load("serizba/salad", "dinov2_salad")
        self.model.eval()
        self.model.cuda()

        self.base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize((644, 644))
            ]
        )

    @method()
    def inference(self, image: bytes, panos: List[bytes], n: int = 5, batch_size=40):
        pano_tensors = []
        for i in range(0, len(panos), batch_size):
            batch = panos[i : i + batch_size]
            with torch.no_grad():
                inp = torch.stack(
                    [
                        self.base_transform(Image.open(BytesIO(i)).convert("RGB"))
                        for i in batch
                    ]
                ).to("cuda")
                batch_tensor = self.model(inp)
                del inp
                pano_tensors.append(batch_tensor)
        pano_tensor = torch.cat(pano_tensors, dim=0)
        img_tensor = self.model(
            torch.stack(
                [self.base_transform(Image.open(BytesIO(image)).convert("RGB"))]
            ).to("cuda")
        )

        cos_sim = cosine_similarity(img_tensor, pano_tensor)
        return cos_sim.tolist()[0]


@stub.local_entrypoint()
def main():
    import os
    import glob

    src_path = ".images/sv_test"
    src_locs = glob.glob(f"{src_path}/db/*.png")
    im_db = [open(loc, "rb").read() for loc in src_locs]
    tar = open(f"{src_path}/targ.png", "rb").read()
    print("Images Loaded.")
    res = VPRModel().inference.remote(tar, im_db)
    print(res)
