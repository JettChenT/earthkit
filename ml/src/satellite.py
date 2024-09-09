import modal
from modal import App
import pickle
from .geo import Coords, Bounds, Point
from .google_map_downloader import download_google_map_area, download_sat_coords
from .rpc import ResultsUpdate

app = App('satellite')

image = (modal
         .Image.debian_slim(python_version="3.11")
         .pip_install_from_pyproject("pyproject.toml"))

@app.function(image=image)
def satellite_locate(target: bytes, bounds: Bounds, zoom = 20):
    res = download_google_map_area(bounds, zoom)
    images = [r.aux['image'] for r in res]
    CrossViewModel = modal.Cls.lookup("crossview", "CrossViewModel")
    prediction_res = CrossViewModel.predict.remote(images, target)
    for (i, r) in enumerate(prediction_res):
        res.coords[i].aux['sim'] = r
        res.coords[i].aux.pop('image', None)
    res.coords.sort(key=lambda x: x.aux['sim'], reverse=True)
    return res

@app.function(image=image)
def satellite_sim(target:bytes, coords:Coords, zoom=20):
    res = download_sat_coords(coords, zoom)
    res.inject_idx()
    images = [r.aux['image'] for r in res]
    CrossViewModel = modal.Cls.lookup("crossview", "CrossViewModel")
    prediction_res = CrossViewModel.predict.remote(images, target)
    coords.inject_idx()
    for (i, r) in enumerate(prediction_res):
        res.coords[i].aux['sim'] = r
    return ResultsUpdate.from_coords(res, 'sim')

@app.local_entrypoint()
def main():
    from .geo import Distance
    res: ResultsUpdate = satellite_sim.remote(open('.images/sf_test.png', 'rb').read(), Bounds.from_points(
        Point(lat=37.789733, lon=-122.402614), Point(lat=37.784409, lon=-122.394974)
    ).sample(Distance(kilometers=0.05)))
    # pickle.dump(res, open('tmp/res.pkl', 'wb'))
    print(res)
