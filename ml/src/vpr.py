import modal
from modal import Secret, Stub, gpu, build, enter, method
from typing import List, Tuple, Any
import asyncio
import os

import modal

stub = Stub("vpr")

inference_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.2.2", "pillow==10.3.0", "torchvision==0.17.2"
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
        _ = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        ).to("cuda")

    @enter()
    def enter(self):
        self.model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        ).to("cuda")

        self.base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @method()
    def inference(self, image: bytes, panos: List[bytes], n: int = 5, batch_size=200):
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

        def _cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
            dot_product = vec1 @ vec2.T
            norm_vec1 = torch.norm(vec1)
            norm_vec2 = torch.norm(vec2, dim=1)
            similarity = dot_product / (norm_vec1 * norm_vec2)
            return similarity

        cos_sim = _cosine_similarity(img_tensor, pano_tensor)
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
