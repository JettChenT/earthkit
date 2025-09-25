import modal
from modal import Secret, App, enter, method
import torch
from torchvision import transforms
from .timmod import TimmModel
from .common import cosine_similarity
from PIL import Image
from io import BytesIO
from typing import List

app = App("crossview")

image = (
    modal.Image
    .debian_slim(python_version="3.10")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("./weights", "/weights", copy=True)
)

@app.cls(gpu="A10G", image=image)
class CrossViewModel:
    @enter()
    def enter(self):
        self.model = TimmModel("convnext_base.fb_in22k_ft_in1k_384", pretrained=True, img_size=384)
        model_state_dict = torch.load("/weights/s4geo_cvusa_weights_e40_98.6830.pth", map_location=torch.device("cuda"))
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(torch.device("cuda"))
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    @method()
    def predict(self, src: List[bytes], target: bytes) -> List[float]:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        src_tensors = torch.stack([preprocess(Image.open(BytesIO(img)).convert("RGB")) for img in src]).to("cuda")
        target_tensor = torch.stack([preprocess(Image.open(BytesIO(target)).convert("RGB"))]).to("cuda")

        with torch.no_grad():
            src_features = self.model(src_tensors)
            target_features = self.model(target_tensor)

        src_features = torch.nn.functional.normalize(src_features, dim=-1)
        target_features = torch.nn.functional.normalize(target_features, dim=-1)

        similarity = cosine_similarity(target_features, src_features)[0]

        return similarity.tolist()

@app.local_entrypoint()
def main():
    from glob import glob
    import pickle
    base_images = [open(img, "rb").read() for img in glob("./tiles/*.png")]
    target_image = open(".images/sf_test.png", "rb").read()
    cvm = CrossViewModel()
    res = cvm.predict.remote(base_images, target_image)
    print(res)
    # pickle.dump(res, open("./tmp/sf_test.pkl", "wb"))
