import modal
import urllib.request
stub = modal.Stub("geoclip")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "pandas==2.2.2",
        "geoclip==1.2.0"
    )
)

@stub.function(image=image, gpu="T4")
def geoclip_inference(image: bytes):
    from geoclip import GeoCLIP
    import io
    model = GeoCLIP().to("cuda")
    top_pred_gps, top_pred_labels = model.predict(io.BytesIO(image), top_k=20)
    return top_pred_gps, top_pred_labels

@stub.local_entrypoint()
def main():
    image = (
        urllib.request
        .urlopen(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/B%C3%B8rsen_1.jpg/242px-B%C3%B8rsen_1.jpg"
        )
        .read()
    )
    print(geoclip_inference.remote(image))
