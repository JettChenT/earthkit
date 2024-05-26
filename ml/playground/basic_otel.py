from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from dotenv import load_dotenv
import asyncio
import random

load_dotenv()
from .otel import tracer_provider

app = FastAPI()

FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)

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
