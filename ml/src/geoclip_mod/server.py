from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .gclip_fast import GeoCLIP
from src.endpoint import GeoclipRequest, get_ip, get_api_key
from src.utils import enforce_im_url
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from src.auth import get_current_user
from src.db import verify_cost, ratelimit
from src.otel import otlp_exporter
from src import schema
from typing import Optional, List
from .gclip_retr import infer_image
import time
import os

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

resource = Resource(attributes={
    "service.name": "geoclip-api"
})

tracer_provider = TracerProvider(
    resource=resource
)
meter_provider = MeterProvider(resource=resource)
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)

app = FastAPI()

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://earthkit.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider, meter_provider=meter_provider)


@app.post("/geoclip")
async def geoclip_inference(
    request: GeoclipRequest,
    user: Optional[str] = Depends(get_current_user),
    request_ip: str = Depends(get_ip),
    api_key: Optional[str] = Depends(get_api_key)
) -> list[schema.Point]:
    if user is None:
        if api_key is None:
            await ratelimit(request_ip)
        elif api_key != os.getenv("API_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
    else:
        await verify_cost(user, 1)

    img = await enforce_im_url(request.image_url)
    assert len(img) < 1000000, "Image url too long"

    t_start_inference = time.time()
    coords, probs = await infer_image(img, request.top_k)
    t_end_inference = time.time()
    print(f"Inference took {t_end_inference - t_start_inference:.4f} seconds")

    pnts: List[schema.Point] = [
        schema.Point(lon=coord[1], lat=coord[0], aux={'pred': prob}) for coord, prob in zip(coords, probs)
    ]

    return pnts

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)