from dotenv import load_dotenv
import os
import time
import httpx
import asyncio
import base64
from io import BytesIO
from src.utils import proc_im_url_async

a_client = httpx.AsyncClient(timeout=1000)
load_dotenv()
REMOTE_URL = "http://localhost:8000/geoclip"
IMG_URL = "https://jld59we6hmprlla0.public.blob.vercel-storage.com/f5452a97-b698-4fac-a92f-6555a270287b-rppzqvZ4jKCgMqm5LMOd6hhcp3FfAz.jpeg"
CONCURRENT_REQUESTS = 5 

async def download_and_encode_image(url):
    image_data = await proc_im_url_async(url)
    base64_image = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')
    return base64_image

async def send_request(img_url: str):
    t_0 = time.time()
    response = await a_client.post(REMOTE_URL, 
                                   json={"image_url": img_url},
                                   headers={"X-API-Key": os.getenv("API_KEY") or ""},
                                   timeout=100000)
    t_1 = time.time()
    return response.json(), t_1 - t_0

async def main():
    start_time = time.time()
    img_b64 = await download_and_encode_image(IMG_URL)
    tasks = [send_request(img_b64) for _ in range(CONCURRENT_REQUESTS)]
    res = await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
    print([res[1] for res in res])

if __name__ == "__main__":
    asyncio.run(main())
