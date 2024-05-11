import modal
from modal import gpu, build, enter, method
from .geoclip_mod.gclip import GeoCLIP
import urllib.request
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
    def inference(self, image: bytes):
        top_pred_gps, top_pred_labels = self.model.predict(io.BytesIO(image), top_k=20)
        return top_pred_gps.tolist(), top_pred_labels.tolist()

@app.local_entrypoint()
def main():
    image = urllib.request.urlopen(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/B%C3%B8rsen_1.jpg/242px-B%C3%B8rsen_1.jpg"
    ).read()
    print("image downloaded")
    print(GeoCLIPModel().inference.remote(image))
