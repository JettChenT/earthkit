from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .gclip import GeoCLIP
from src.endpoint import GeoclipRequest, get_ip, proc_im_url_async, get_api_key
from src.auth import get_current_user
from src.db import verify_cost, ratelimit
from src import schema
from typing import Optional, List
import httpx
import time
import io
import os

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

print("Starting server... loading GeoCLIP model")
t_start = time.time()
model = GeoCLIP()
t_end = time.time()
print(f"Server started in {t_end - t_start} seconds")

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

    img = await proc_im_url_async(request.image_url)

    t_start_inference = time.time()
    res_gps, res_pred = model.predict(io.BytesIO(img), request.top_k)
    t_end_inference = time.time()
    print(f"Inference took {t_end_inference - t_start_inference:.4f} seconds")

    res_gps, res_pred = res_gps.tolist(), res_pred.tolist()
    pnts : List[schema.Point] = [
        schema.Point(lon=gps[1], lat=gps[0], aux={'pred':pred}) for gps, pred in zip(res_gps, res_pred)
    ]

    return pnts


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)