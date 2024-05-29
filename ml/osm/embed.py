from openai import embeddings
from dotenv import load_dotenv

load_dotenv()

queries = open("osm/dataset/dataset.train.query").read().splitlines()

print(len(queries))
emb = embeddings.create()