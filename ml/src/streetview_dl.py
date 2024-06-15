import asyncio
from .otel import instrument, OTEL_DEPS, get_tracer, ENVS
instrument()

import modal
from modal import Stub
from typing import List
from .streetview import search_panoramas, get_panorama_async
from .geo import Coords, Point, Bounds, Distance
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from aiostream import stream, pipe, streamcontext
from .stream_utils import eager_chunks
from .rpc import ProgressUpdate
import time

stub = Stub("streetview-locate")

sv_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "geojson==3.1.0", "streetview==0.0.6", "geopy==2.4.1", "aiostream==0.5.2", "httpx==0.27.0", "cattrs==23.2.3", *OTEL_DEPS
).env(ENVS)


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

M_TOP = 100
M_BOTTOM = 180
NUM_DIR = 6

def crop_pano(pano: Image.Image, n_img=NUM_DIR) -> List[Image.Image]:
    panorama = pano.crop((0, M_TOP, pano.width, pano.height - M_BOTTOM))
    wid, hei = panorama.size
    num_crops = n_img

    cropped_images = []
    stride = (wid - hei) / (num_crops - 1)
    for i in range(num_crops):
        left = i * stride
        right = left + hei
        crop_box = (left, 0, right, hei)
        cropped_img = panorama.crop(crop_box)
        cropped_images.append(cropped_img)
    return cropped_images


@stub.function(image=sv_image)
async def streetview_locate(panos: Coords, image: bytes, inference_batch_size=360, download_only=False):
    tracer = get_tracer()

    @tracer.start_as_current_span("fetch_image")
    async def fetch_image(pano: Point):
        pid = pano.aux["pano_id"]
        pano_im = await get_panorama_async(pid, zoom=2)
        crops = crop_pano(pano_im)
        return (pid, [{
            "image": cropped, 
            "dir": dir
        } for dir, cropped in enumerate(crops)])

    pid_map = {p.aux['pano_id']: p for p in panos.coords}

    print("fetching streetviews...")
    VPRModel = modal.Cls.lookup("vpr", "VPRModel")
    inference = VPRModel().inference

    async def inference_batch(batch):
        flattened = [im['image'] for record in batch for im in record[1]]
        res = await inference.remote.aio(image, flattened)
        updated_pnts = []
        for i in range(0, len(batch), NUM_DIR):
            chunk = res[i:i+NUM_DIR]
            batch_i = i//NUM_DIR
            pid = batch[batch_i][0]
            pnt = pid_map[pid]
            pnt.aux.update({
                "sims": chunk,
                "max_sim": max(chunk),
                "max_sim_ind": chunk.index(max(chunk))
            })
            updated_pnts.append(pnt)
        return Coords(updated_pnts)
            
    num_cords = len(panos.coords)
    coords_iter = stream.iterate(panos.coords)
    xs = coords_iter | pipe.map(fetch_image, ordered=False)
    batch_cnt = inference_batch_size // NUM_DIR
    inference_queue = asyncio.Queue()
    response_queue = asyncio.Queue()

    async def task_download_panos():
        async with streamcontext(xs) as streamer:
            cur_batch = []
            async for result in streamer:
                await response_queue.put(ProgressUpdate(
                    "Streetview Panorama Fetched",
                    total=num_cords,
                    current=[result[0]]
                ))
                cur_batch.append(result)
                if len(cur_batch) == batch_cnt:
                    await inference_queue.put(asyncio.create_task(inference_batch(cur_batch)))
                    cur_batch = []
            if cur_batch:
                await inference_queue.put(asyncio.create_task(inference_batch(cur_batch)))
            await inference_queue.put(None)
        print("all panoramas fetched")
    
    pano_download_task = asyncio.create_task(task_download_panos())
    

    async def run_inference_q():
        while True:
            task = await inference_queue.get()
            if task is None:
                break
            res = await task
            await response_queue.put(res)
            inference_queue.task_done()
        await response_queue.put(None)
    
    inference_task = asyncio.create_task(run_inference_q())

    while True:
        res = await response_queue.get()
        if res is None:
            break
        yield res
    


@stub.local_entrypoint()
async def main():
    import dotenv
    import pickle
    USE_SMP_CACHED = True
    LIMIT_CNT = 100
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
    if LIMIT_CNT is not None:
        sampled_views = Coords(sampled_views.coords[:LIMIT_CNT])
    print(f"sampled {len(sampled_views)} streetviews")
    print(sampled_views[:30])
    im = open("tmp/fsr.png", "rb").read()
    async for res in streetview_locate.remote_gen.aio(sampled_views, im):
        print(res)
    # proced_views: Coords = streetview_locate.remote(sampled_views, im)
    # print("I'mmmmmm FINISHHHHHHHHED")
    # print(f"fetched {len(proced_views)} streetviews")
    # pickle.dump(proced_views, open("tmp/proced_views_ml.pkl", "wb"))
    # print(proced_views[:30])
    # sampled_views.plot("tmp/sampled.html")
    # proced_views[:30].plot("tmp/processed.html")
