import modal
from modal import Secret, Stub
from typing import List, Tuple
from .streetview import search_panoramas, get_streetview, get_panorama
from concurrent.futures import ThreadPoolExecutor
from .geo import Coords, Point, Bounds, Distance
from equilib import equi2pers
import numpy as np
from io import BytesIO
import os

import modal

stub = Stub("streetview-locate")

sv_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "geojson==3.1.0", "streetview==0.0.6", "geopy==2.4.1", "pyequilib==0.5.8", "torch"
)


def fetch_panoramas(coord: Point):
    res = {}
    for pno in search_panoramas(coord.lat, coord.lon):
        res.update(
            {
                pno.pano_id: {
                    "pano_id": pno.pano_id,
                    "lat": pno.lat,
                    "lon": pno.lon,
                    "date": pno.date,
                }
            }
        )
    return res


@stub.function(image=sv_image)
def sample_streetviews(bounds: Bounds, interval: Distance):
    sampled = bounds.sample(interval)
    pnts = {}
    print("getting panoramas")
    with ThreadPoolExecutor() as executor:
        res = list(executor.map(fetch_panoramas, sampled))
    for pano in res:
        pnts.update(pano)
    print("getting panoramas done")
    coords = Coords([Point(pnt["lon"], pnt["lat"], pnt) for pnt in pnts.values()])
    w, h = bounds.get_wh()
    w_len, h_len = int(w.km / interval.km), int(h.km / interval.km)
    w_co, h_co = (
        (bounds.hi.lon - bounds.lo.lon) / w_len,
        (bounds.hi.lat - bounds.lo.lat) / h_len,
    )
    print(w, h, w_len, h_len, w_co, h_co)
    mtrix: List[None | Point] = [None for _ in range(w_len * h_len * 2)]

    def parse_date(s):
        return list(map(int, s.split("-"))) if s else [0, 0]

    for pnt in coords:
        x, y = (
            int((pnt.lat - bounds.lo.lat) // h_co),
            int((pnt.lon - bounds.lo.lon) // w_co),
        )
        # print(pnt, x, y, x*w_len + y)
        cur = mtrix[x * w_len + y]
        if (
            cur is None
            or cur.aux["date"] is None
            or parse_date(cur.aux["date"]) < parse_date(pnt.aux["date"])
        ):
            mtrix[x * w_len + y] = pnt
    return Coords(list(filter(lambda x: x, mtrix)))


@stub.function(image=sv_image)
def streetview_locate(panos: Coords, image: bytes, batch_size=400, download_only=False):
    images = []

    def fetch_image(args):
        i, pano = args
        buf = BytesIO()
        pano = get_panorama(pano.aux["pano_id"], zoom=2, multi_threaded=True)
        # return (i, [{
        #     "image": pano, 
        #     "dir": 0
        # }])
        pano = np.transpose(np.asarray(pano), (2,0,1))
        res = []
        for dir in range(4):
            pers = equi2pers(
                equi=pano,
                rots = {
                    'roll':0,
                    'pitch':0,
                    'yaw': np.pi/2*dir
                },
                height=640,
                width=640,
                fov_x=120,
            )
            res.append({
                "image": np.transpose(pers, (1,2,0)),
                "dir": dir*90
            })
        return (i, res)

    print("fetching streetviews...")
    with ThreadPoolExecutor() as executor:
        images = list(
            executor.map(
                fetch_image,
                [
                    (i, pano)
                    for i, pano in enumerate(panos)
                ],
            )
        )
        images = [(i, im['image'], im['dir']) for i, ims in images for im in ims]
    print(f"fetched {len(images)} streetviews, running VPR...")
    if download_only:
        return images
    VPRModel = modal.Cls.lookup("vpr", "VPRModel")
    batched_images = [im[1] for im in images]
    similarity_batches = VPRModel().inference.starmap(
        [
            (image, batched_images[i : i + batch_size])
            for i in range(0, len(batched_images), batch_size)
        ]
    )
    similarity = [sim for batch in similarity_batches for sim in batch]
    for i, sim in enumerate(similarity):
        cord_i = images[i][0]
        cord = panos[cord_i]
        if "max_sim" not in cord.aux:
            cord.aux["max_sim"] = sim
            cord.aux["max_sim_ind"] = 0
            cord.aux["sims"] = [{"dir": images[i][2], "sim": sim, "index": i}]
        else:
            cord.aux["max_sim"] = max(cord.aux["max_sim"], sim)
            if sim > cord.aux["max_sim"]:
                cord.aux["max_sim_ind"] = len(cord.aux["sims"])
            cord.aux["sims"].append({"dir": images[i][2], "sim": sim, "index": i})
    panos.coords.sort(key=lambda x: x.aux["max_sim"], reverse=True)
    return panos


@stub.local_entrypoint()
def main():
    import dotenv
    import pickle
    USE_SMP_CACHED = True
    dotenv.load_dotenv()
    if USE_SMP_CACHED:
        sampled_views = pickle.load(open("tmp/sampled_views.pkl", "rb"))
    else:
        bounds = Bounds.from_points(
            Point(lat=37.789733, lon=-122.402614), Point(lat=37.784409, lon=-122.394974)
        )
        interval = Distance(kilometers=0.03)
        sampled_views: Coords = sample_streetviews.remote(bounds, interval)
        pickle.dump(sampled_views, open("tmp/sampled_views.pkl", "wb"))
    print(f"sampled {len(sampled_views)} streetviews")
    print(sampled_views[:30])
    im = open("tmp/fsr.png", "rb").read()
    proced_views: Coords = streetview_locate.remote(sampled_views, im, download_only=True)
    print(f"fetched {len(proced_views)} streetviews")
    pickle.dump(proced_views, open("tmp/proced_views.pkl", "wb"))
    # print(proced_views[:30])
    # sampled_views.plot("tmp/sampled.html")
    # proced_views[:30].plot("tmp/processed.html")
