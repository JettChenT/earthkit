"""
This script precomputes all location embeddings and dumps them to a vector index
"""

from upstash_vector import Index
from .gclip_fast import GeoCLIP
import os
from tqdm import tqdm

import os
from dotenv import load_dotenv

if not os.getenv("UPSTASH_VECTOR_REST_URL") or not os.getenv("UPSTASH_VECTOR_REST_TOKEN"):
    load_dotenv()

index = Index(url=os.environ["UPSTASH_VECTOR_REST_URL"], token=os.environ["UPSTASH_VECTOR_REST_TOKEN"])
gclip_mod = GeoCLIP()
loc_embeddings = gclip_mod.location_feats.cpu().detach().numpy()
coords = gclip_mod.gps_gallery.cpu().detach().numpy()

BATCH_SIZE = 1000
for i in tqdm(range(0, len(loc_embeddings), BATCH_SIZE)):
    batch_embeddings = loc_embeddings[i:i+BATCH_SIZE]
    batch_coords = coords[i:i+BATCH_SIZE]
    dat = [
        (str(i*BATCH_SIZE+j), emb, {"lon": float(lon), "lat": float(lat)}) for j, (emb, (lat, lon)) in enumerate(zip(batch_embeddings, batch_coords))
    ]
    index.upsert(
        vectors=dat,
    )
