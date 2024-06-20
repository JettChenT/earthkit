from ..schema import Coords
from ..geo import Point
from ..streetview import fetch_single_pano
from ..rpc import ResultsUpdate, SiftResult
from ..streetview import get_panorama_async
from ..google_map_downloader import download_point_async
from ..utils import pil_im_url
from .prompting import SYS_PROMPT
from .mm_utils import render_text_description, encode_image

from pydantic import BaseModel
from enum import Enum
from typing import List, Optional
import asyncio
from aiostream import stream, pipe
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
import traceback
import os

class Dependency(BaseModel):
    satellite: bool
    streetview: bool
    basicmap: bool
    target_image: bool

class OutputEnum(str, Enum):
    text = "text"
    number = "number"
    boolean = "boolean"

class LMMConfig(BaseModel):
    model: str = "gpt-4o"

class LmmRequest(BaseModel):
    dependencies: Dependency
    prompt: str
    output_type: OutputEnum
    coords: Coords
    target_image: Optional[str] = None
    config: LMMConfig


def process_response(res:str, format: OutputEnum):
    thought_process = '\n'.join(res.splitlines()[:-1])
    final_content = res.splitlines()[-1].strip()
    if format == OutputEnum.text:
        return final_content, thought_process
    elif format == OutputEnum.number:
        return float(final_content), thought_process
    elif format == OutputEnum.boolean:
        return final_content.lower()[:3] == "yes", thought_process

async def process_request(req: LmmRequest):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
    async def fetch_dependencies(point:Point) -> List[ChatCompletionMessageParam]:
        tasks = []
        if req.dependencies.streetview:
            async def sv_task():
                pano_id = await fetch_single_pano(point.lat, point.lon)
                if not pano_id:
                    return None
                res_im = await get_panorama_async(pano_id, 2)
                return render_text_description(res_im, "Streetview Panorama")
            tasks.append(asyncio.create_task(sv_task()))
        if req.dependencies.satellite:
            async def sat_task():
                res = await download_point_async(point)
                return render_text_description(res, "Satellite Image")
            tasks.append(asyncio.create_task(sat_task()))
        if req.dependencies.basicmap:
            async def bm_task():
                res = await download_point_async(point, lyr="m")
                return render_text_description(res, "Basic Map")
            tasks.append(asyncio.create_task(bm_task()))
        await asyncio.gather(*tasks)
        res = [task.result() for task in tasks]
        if req.dependencies.target_image:
            if not req.target_image:
                raise ValueError("Target image is required")
            res.append(render_text_description(pil_im_url(req.target_image), "Target Image"))
        return [{"role": "user", "content": [
            {"type":"image_url", "image_url":{"url": encode_image(im)}} for im in res if im
        ]}]

    async def proc_point(idx:int, point: Point):
        try:
            deps = await fetch_dependencies(point)
            messages = [{
                "role": "system",
                "content": SYS_PROMPT
            }, *deps, {
                "role": "user",
                "content": f"{req.prompt}"
            }]
            res = await client.chat.completions.create(
                model=req.config.model,
                messages=messages
            )
            content = res.choices[0].message.content
            if not content:
                raise ValueError("No content")
            ans, thought = process_response(content, req.output_type)
            return ResultsUpdate([SiftResult(idx, {
                "answer": ans,
                "thought": thought
            })])
        except Exception as e:
            print(e)
            traceback.print_exc()
            return ResultsUpdate([SiftResult(idx, None, str(e))])
    coords = req.coords.to_geo()
    xs = stream.iterate(coords.coords) | pipe.enumerate() | pipe.starmap(proc_point)
    async with xs.stream() as streamer:
        async for x in streamer:
            yield x