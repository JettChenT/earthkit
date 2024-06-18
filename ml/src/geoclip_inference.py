import modal
from modal import gpu, build, enter, method
from .geoclip_mod.gclip import GeoCLIP
from .geo import Coords
import urllib.request
import torch
import io

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.2.2", "pandas==2.2.2", "geoclip==1.2.0"
)
app = modal.App("geoclip", image=image)

@app.cls(
    gpu=gpu.A10G(),
)
class GeoCLIPModel:
    @build()
    def build(self):
        _ = GeoCLIP()
    
    @enter()
    def setup(self):
        self.model = GeoCLIP()
        self.model = self.model.to("cuda")
    
    @method()
    def inference(self, image: bytes, top_k=100, poke=False):
        if poke:
            return
        top_pred_gps, top_pred_labels = self.model.predict(io.BytesIO(image), top_k=top_k)
        return top_pred_gps.tolist(), top_pred_labels.tolist()
    
    @method()
    def similarity(self, image: bytes, coords: Coords) -> Coords:
        gps_gallary = list(map(lambda x: (x.lat, x.lon), coords.coords))
        gps_gallary = torch.tensor(gps_gallary)
        probs_per_image = self.model.predict_coords(io.BytesIO(image), gps_gallary).tolist()
        for i, prob in enumerate(probs_per_image):
            coords.coords[i].update_aux(geoclip_sim=prob)
        return coords

@app.local_entrypoint()
def main():
    image = urllib.request.urlopen(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/B%C3%B8rsen_1.jpg/242px-B%C3%B8rsen_1.jpg"
    ).read()
    print("image downloaded")
    print(GeoCLIPModel().inference.remote(image))
