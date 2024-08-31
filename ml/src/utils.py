import base64
from fastapi import HTTPException
import requests
from fastapi.encoders import jsonable_encoder
from PIL import Image
from io import BytesIO
from httpx import AsyncClient
import json
from .vercel_blob import put
from uuid import uuid4
import filetype

aclient = AsyncClient()

def proc_im_url(image_url: str) -> bytes:
    if image_url.startswith('data:'):
        base64_str_start = image_url.find('base64,') + 7
        image_data = base64.b64decode(image_url[base64_str_start:])
        return image_data
    else:
        response = requests.get(image_url)
        response.raise_for_status() 
        return response.content

async def proc_im_url_async(image_url: str) -> bytes:
    try:
        if image_url.startswith('data:'):
            base64_str_start = image_url.find('base64,') + 7
            image_data = base64.b64decode(image_url[base64_str_start:])
            return image_data
        else:
            print(f"Downloading image from {image_url}")
            response = await aclient.get(image_url)
            response.raise_for_status() 
            return response.content
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

async def enforce_im_url(image_url: str) -> str:
    try:
        if image_url.startswith('data:'):
            data = proc_im_url(image_url)
            kind = filetype.guess(data)
            if kind is None:
                ext = "bin"
                mime = "application/octet-stream"
            else:
                ext = kind.extension
                mime = kind.mime
            url = f'earthkit_uploads/{uuid4()}.{ext}'
            resp = await put(url, data, mime)
            return resp['url']
        else:
            return image_url
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

def pil_im_url(image_url: str) -> Image.Image:
    return Image.open(BytesIO(proc_im_url(image_url)))

def json_encode(obj) -> str:
    return json.dumps(jsonable_encoder(obj))