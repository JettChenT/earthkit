import base64
import requests
from fastapi.encoders import jsonable_encoder
from PIL import Image
from io import BytesIO
import json

def proc_im_url(image_url: str) -> bytes:
    if image_url.startswith('data:'):
        base64_str_start = image_url.find('base64,') + 7
        image_data = base64.b64decode(image_url[base64_str_start:])
        return image_data
    else:
        response = requests.get(image_url)
        response.raise_for_status() 
        return response.content

def pil_im_url(image_url: str) -> Image.Image:
    return Image.open(BytesIO(proc_im_url(image_url)))

def json_encode(obj) -> str:
    return json.dumps(jsonable_encoder(obj))