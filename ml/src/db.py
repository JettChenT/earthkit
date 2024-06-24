import os
from attr import define
from fastapi import HTTPException
import redis.asyncio as redis

DEFAULT_CREDIT=500

async def get_client():
    return await redis.Redis(
        host=os.getenv("REDIS_HOST") or "localhost",
        port=int(os.getenv("REDIS_PORT") or 6379),
        password=os.getenv("REDIS_PASSWORD"),
        ssl=True
    )

@define
class UsageData:
    quota: int
    remaining: int

async def get_usage(user:str):
    client = await get_client()
    remaining, quota = await client.hmget(user, "remaining", "quota") 
    if remaining is None:
        remaining = quota = DEFAULT_CREDIT
        await client.hmset(user, {"remaining": str(remaining), "quota": str(DEFAULT_CREDIT)})
    return UsageData(quota=int(quota), remaining=int(remaining))

async def verify_cost(user:str, cost:int):
    assert cost >= 0
    client = await get_client()
    remaining = await client.hincrby(user, "remaining", -cost)
    if remaining == -cost and (await client.hget(user, "quota")) is None:
        await client.hmset(user, {
            "remaining": DEFAULT_CREDIT - cost,
            "quota": DEFAULT_CREDIT,
        })
        remaining = DEFAULT_CREDIT - cost
    if remaining < 0:
        await client.hincrby(user, "remaining", cost)
        raise HTTPException(
            status_code=402,
            detail=f"Quota exceeded. This operation requires a credit of at least {cost}. You have {remaining + cost} credits remaining."
        )
    return remaining