"""
Vector DB retrieval based GeoCLIP
"""

import torch
import torch.nn.functional as F
from upstash_vector import AsyncIndex
from .tsfm import Tsfm
import replicate
import os
import asyncio

EXP = 39.6874
tsfm = Tsfm()
tsfm.eval()
tsfm.load_weights()

async def infer_image(img_url: str, top_k: int = 100):
    clip_embs = await replicate.async_run(
        "andreasjansson/clip-features:75b33f253f7714a281ad3e9b28f63e3232d583716ef6718f2e46641077ea040a",
        input={"inputs": img_url}
    )

    emb_tensor = torch.tensor(clip_embs[0]['embedding']).unsqueeze(0)

    with torch.no_grad():
        img_feats = F.normalize(tsfm(emb_tensor))
        res_vec = img_feats.squeeze(0).tolist()

    idx = AsyncIndex(url=os.environ["UPSTASH_VECTOR_REST_URL"], token=os.environ["UPSTASH_VECTOR_REST_TOKEN"])

    res = await idx.query(
        vector=res_vec,
        top_k=top_k,
        include_metadata=True,
    )

    coords = [(item.metadata.get('lat'), item.metadata.get('lon')) for item in res]
    with torch.no_grad():
        probs = [item.score*EXP for item in res]
        probs = torch.tensor(probs)
        probs = F.softmax(probs, dim=0).tolist()
    return coords, probs

async def main():
    tst_img = "https://jld59we6hmprlla0.public.blob.vercel-storage.com/1d43cbec-f01c-4e21-bf17-4d455a69974e-l1HNRCEvZQ2nDDgc36cUTmiNGGeCms.png"
    coords, probs = await infer_image(tst_img)
    print(coords, probs)

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    asyncio.run(main())