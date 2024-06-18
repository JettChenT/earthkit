import modal
from modal import gpu, build, enter, method
from .geoclip_mod.gclip import GeoCLIP
from .geo import Coords
from .rpc import ResultsUpdate, SiftResult
import urllib.request
import torch
import io

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.2.2", "pandas==2.2.2", "geoclip==1.2.0", "cattrs==23.2.3"
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
    def similarity(self, image: bytes, coords: Coords) -> ResultsUpdate:
        gps_gallary = list(map(lambda x: (x.lat, x.lon), coords.coords))
        gps_gallary = torch.tensor(gps_gallary)
        probs_per_image = self.model.predict_coords(io.BytesIO(image), gps_gallary).tolist()
        results = []
        for i, prob in enumerate(probs_per_image[0]):
            results.append(SiftResult(idx=i, value=prob))
        return ResultsUpdate(results=results)

@app.local_entrypoint()
def main():
    from .geo import Bounds, Point, Distance
    image = urllib.request.urlopen(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/B%C3%B8rsen_1.jpg/242px-B%C3%B8rsen_1.jpg"
    ).read()
    print("image downloaded")
    bounds = Bounds(
        Point(lat=37.789733, lon=-122.402614), Point(lat=37.784409, lon=-122.394974)
    )
    interval = Distance(kilometers=0.05)
    sampled = bounds.sample(interval)
    # print(GeoCLIPModel().inference.remote(image))
    print(GeoCLIPModel().similarity.remote(image, sampled))
