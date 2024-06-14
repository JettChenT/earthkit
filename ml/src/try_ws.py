from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel

class NumberRequest(BaseModel):
    n: int

@app.post("/dummy-sse")
async def dummy_sse(request: NumberRequest):
    async def event_generator():
        import json
        import asyncio
        for i in range(request.n):
            await asyncio.sleep(1)
            yield f"data: {json.dumps({'message': f'Event {i}; {request.n}'})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
