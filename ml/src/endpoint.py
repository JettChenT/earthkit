from modal import App, asgi_app, Secret
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import modal
from typing import List, Optional
from . import schema, geo
from .utils import json_encode, proc_im_url, proc_im_url_async
from .rpc import ResultsUpdate, encode_msg, sse_encode
from .otel import tracer_provider, meter_provider
from .cfig import ENVS
from . import lmm
from .auth import get_current_user
import math
from .db import verify_cost, get_usage, ratelimit
import cattrs
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from typing import Any
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi.openapi.utils import get_openapi
import os
import sentry_sdk

image = (modal
         .Image
         .debian_slim(python_version="3.11")
         .pip_install_from_pyproject("pyproject.toml")
         .env(ENVS))

if os.getenv("SENTRY_DSN"):
    sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"))

web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FastAPIInstrumentor.instrument_app(web_app, tracer_provider=tracer_provider, meter_provider=meter_provider)


def get_ip(request: Request):
    return request.client.host

def get_api_key(request: Request):
    return request.headers.get("X-API-Key")

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
    coords: schema.Coords[Any]
    image_url: str

@web_app.post("/streetview/locate/streaming")
async def streetview_locate_sse(request: SVLocateRequest, user: str = Depends(get_current_user)):
    await verify_cost(user, math.ceil(len(request.coords.coords)/60))
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
async def geoclip_inference(request: GeoclipRequest, user: Optional[str] = Depends(get_current_user), request_ip: str = Depends(get_ip)) -> list[schema.Point]:
    if user is None:
        await ratelimit(request_ip)
    else:
        await verify_cost(user, 1)
    c = modal.Cls.lookup("geoclip", "GeoCLIPModel")
    print("downloading image...")
    img = await proc_im_url_async(request.image_url)
    print("running inference...")
    res_gps, res_pred = await c.inference.remote.aio(img, request.top_k)
    pnts : List[schema.Point] = [
        schema.Point(lon=gps[1], lat=gps[0], aux={'pred':pred}) for gps, pred in zip(res_gps, res_pred)
    ]
    return pnts

class GeoclipSimilarityRequest(BaseModel):
    image_url: str
    coords: schema.Coords[Any]

@web_app.post("/geoclip/similarity")
async def geoclip_similarity(request: GeoclipSimilarityRequest, user: str = Depends(get_current_user)):
    await verify_cost(user, 1)
    c = modal.Cls.lookup("geoclip", "GeoCLIPModel")
    img = await proc_im_url_async(request.image_url)
    res = await c.similarity.remote.aio(img, request.coords.to_geo())
    return encode_msg(res)

@web_app.post("/geoclip/similarity/streaming")
async def geoclip_similarity_sse(request: GeoclipSimilarityRequest, user: str = Depends(get_current_user)):
    await verify_cost(user, 1)
    async def event_generator():
        c = modal.Cls.lookup("geoclip", "GeoCLIPModel")
        img = await proc_im_url_async(request.image_url)
        coords_geo = request.coords.to_geo()
        res = c.similarity.remote(img, coords_geo)
        yield f"data: {json_encode(encode_msg(res))}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@web_app.get("/geoclip/poke")
def geoclip_poke():
    c = modal.Cls.lookup("geoclip", "GeoCLIPModel")
    c.inference.remote(None, poke=True)
    return {"ok": True}

class SatelliteLocateRequest(BaseModel):
    bounds: schema.Bounds[None]
    image_url: str

@web_app.post("/satellite/locate")
async def satellite_locate(request: SatelliteLocateRequest, user: str = Depends(get_current_user)):
    await verify_cost(user, 1)
    f = modal.Function.lookup("satellite", "satellite_locate")
    img = await proc_im_url_async(request.image_url)
    res: geo.Coords = f.remote(img, request.bounds.to_geo())
    return schema.Coords.from_geo(res)

class SatelliteSimRequest(BaseModel):
    image_url: str
    coords: schema.Coords[Any]

@web_app.post("/satellite/similarity/streaming")
async def satellite_sim_sse(request: SatelliteSimRequest, user: str = Depends(get_current_user)):
    await verify_cost(user, 1)
    async def event_generator():
        f = modal.Function.lookup("satellite", "satellite_sim")
        img = await proc_im_url_async(request.image_url)
        res: ResultsUpdate = f.remote(img, request.coords.to_geo())
        yield sse_encode(res)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@web_app.post("/lmm/streaming")
async def lmm_streaming(request: lmm.LmmRequest, user: str = Depends(get_current_user)):
    await verify_cost(user, math.ceil(len(request.coords.coords)/20))
    async def event_generator():
        async for res in lmm.process_request(request):
            yield sse_encode(res)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


class OrienterNetLocateRequest(BaseModel):
    image_url: str
    location_prior: schema.Point[Any]

@web_app.post("/orienternet/locate")
async def orienternet_locate(request: OrienterNetLocateRequest, user: Optional[str] = Depends(get_current_user), request_ip: str = Depends(get_ip)):
    if user:
        await verify_cost(user, 1)
    else:
        await ratelimit(request_ip)
    c = modal.Cls.lookup("orienternet", "OrienterNetModel")
    res: geo.Point = await c.locate.remote.aio(request.image_url, request.location_prior.to_geo())
    return schema.Point.from_geo(res)

@web_app.get("/test/echo-user")
async def test_user(user: Optional[str] = Depends(get_current_user), ip=Depends(get_ip)):
    if user is None:
        await ratelimit(ip)
        return {"user": None}
    import time
    t0 = time.time()
    usage = await get_usage(user)
    t1 = time.time()
    return {"user": user, "usage": cattrs.unstructure(usage), "usage_fetch_duration": t1-t0}

@web_app.post("/test/simulate_cost")
async def test_simulate_cost(cost:int=1, user: str = Depends(get_current_user)):
    assert cost >= 0
    import time
    t0 = time.time()
    await verify_cost(user, cost)
    t1 = time.time()
    return {"ok": True, "duration": t1-t0}

@web_app.post("/test/simulate_error")
async def test_simulate_error():
    x = 1/0

@web_app.get("/ping")
async def ping():
    return {"message": "pong"}


@web_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(content={"detail": str(exc)}, status_code=422)

def custom_openapi():
    if web_app.openapi_schema:return web_app.openapi_schema
    openapi_schema = get_openapi(title="Earthkit API", version="0.1.0", routes=web_app.routes)
    openapi_schema["components"]["schemas"]["HTTPValidationError"] = {"title": "HTTPValidationError", "type": "object", "properties": {"detail": {"type": "string"}}}
    web_app.openapi_schema = openapi_schema
    return web_app.openapi_schema

web_app.openapi = custom_openapi

@app.function(image=image, secrets=[Secret.from_name("oai"), Secret.from_name("earthkit-backend")])
@asgi_app()
def fastapi_app():
    return web_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000)