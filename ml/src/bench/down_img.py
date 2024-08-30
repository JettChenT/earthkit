from src.utils import proc_im_url_async
import asyncio

async def main():
    img_url = "https://jld59we6hmprlla0.public.blob.vercel-storage.com/f5452a97-b698-4fac-a92f-6555a270287b-rppzqvZ4jKCgMqm5LMOd6hhcp3FfAz.jpeg"
    img_res = await proc_im_url_async(img_url)
    # print(img_b64)
    print(len(img_res))

if __name__ == "__main__":
    asyncio.run(main())