import modal
from modal import Stub
import pickle
from .geo import Coords, Bounds, Point
from .google_map_downloader import download_google_map_area

stub = Stub('satellite')

image = (modal
         .Image.debian_slim(python_version="3.11")
         .pip_install("pillow", "geopy", "tqdm"))

@stub.function(image=image)
def satellite_locate(target: bytes, bounds: Bounds, zoom = 20):
    res = download_google_map_area(bounds, zoom)
    images = [r.aux['image'] for r in res]
    CrossViewModel = modal.Cls.lookup("crossview", "CrossViewModel")
    prediction_res = CrossViewModel().predict.remote(images, target)
    for (i, r) in enumerate(prediction_res):
        res[i].aux['sim'] = r
    return res

@stub.local_entrypoint()
def main():
    res: Coords = satellite_locate.remote(open('.images/sf_test.png', 'rb').read(), Bounds.from_points(
        Point(lat=37.789733, lon=-122.402614), Point(lat=37.784409, lon=-122.394974)
    ))
    pickle.dump(res, open('tmp/res.pkl', 'wb'))
