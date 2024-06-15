from modal import App, asgi_app
from fastapi import FastAPI, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model, ValidationError
import modal
from typing import List, Type, TypeVar
from . import schema, geo
from .utils import json_encode, proc_im_url
from .rpc import encode_msg
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "geopy==2.4.1", "requests==2.28.1", "websockets==12.0", "cattrs==23.2.3"
)

web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = App("ek-endpoint")

class SampleStreetviewsRequest(BaseModel):
    bounds: schema.Bounds[None]
    dist_km: float                              

@web_app.post("/streetview/sample")
def sample_streetviews(request: SampleStreetviewsRequest):
    from geopy.distance import Distance
    f = modal.Function.lookup("streetview-locate", "sample_streetviews")
    res: geo.Coords = f.remote(request.bounds.to_geo(), Distance(request.dist_km))
    return schema.Coords.from_geo(res)

class PanoAux(BaseModel):
    pano_id: str

class SVLocateRequest(BaseModel):
    coords: schema.Coords[PanoAux]
    image_url: str

@web_app.post("/streetview/locate")
async def streetview_locate(request: SVLocateRequest):
    results = []
    res: geo.Coords = results[0]
    return schema.Coords.from_geo(res)

@web_app.post("/streetview/locate/streaming")
async def streetview_locate_sse(request: SVLocateRequest):
    async def event_generator():
        f = modal.Function.lookup("streetview-locate", "streetview_locate")
        img = proc_im_url(request.image_url)
        cords_geo = request.coords.to_geo()
        async for res in f.remote_gen.aio(cords_geo, img):
            yield f"data: {json_encode(encode_msg(res))}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

class GeoclipRequest(BaseModel):
    image_url: str
    top_k: int = 100

@web_app.post('/geoclip')
def geoclip_inference(request: GeoclipRequest):
    c = modal.Cls.lookup("geoclip", "GeoCLIPModel")
    print("downloading image...")
    img = proc_im_url(request.image_url)
    print("running inference...")
    res_gps, res_pred = c.inference.remote(img, request.top_k)
    pnts : List[schema.Point] = [
        schema.Point(lon=gps[0], lat=gps[1], aux={'pred':pred}) for gps, pred in zip(res_gps, res_pred)
    ]
    return pnts

@web_app.get("/geoclip/poke")
def geoclip_poke():
    c = modal.Cls.lookup("geoclip", "GeoCLIPModel")
    c.inference.remote(None, poke=True)
    return {"ok": True}

class SatelliteLocateRequest(BaseModel):
    bounds: schema.Bounds[None]
    image_url: str

@web_app.post("/satellite/locate")
def satellite_locate(request: SatelliteLocateRequest):
    f = modal.Function.lookup("satellite", "satellite_locate")
    img = proc_im_url(request.image_url)
    res: geo.Coords = f.remote(img, request.bounds.to_geo())
    return schema.Coords.from_geo(res)


@app.function(image=image)
@asgi_app()
def fastapi_app():
    return web_app
