from fastapi import FastAPI
from dotenv import load_dotenv
import asyncio
import random

load_dotenv()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, world!"}

@app.get("/data")
async def data():
    return {"data": "Sample data"}

@app.get("/takes_time")
async def takes_time():
    await asyncio.sleep(0.7 + random.random() * 2)
    return {"message": "Hello, world!"}
