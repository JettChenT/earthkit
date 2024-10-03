import os
from attr import define
from fastapi import HTTPException
from .auth import credentials_exception
from upstash_redis.asyncio import Redis
from upstash_ratelimit.asyncio import Ratelimit, FixedWindow

DEFAULT_CREDIT=500

async def get_client():
    return Redis.from_env()

async def get_anon_ratelimit(expensive:bool=False):
    if expensive:
        return Ratelimit(
            await get_client(),
            FixedWindow(5, 60*60),
            prefix="earthkit-anon-ratelimit-expensive"
        )
    return Ratelimit(
        await get_client(),
        FixedWindow(20, 60*60),
        prefix="earthkit-anon-ratelimit"
    )


@define
class UsageData:
    quota: int
    remaining: int

async def get_usage(user:str):
    if user is None:
        raise credentials_exception
    client = await get_client()
    remaining, quota = await client.hmget(user, "remaining", "quota") 
    if remaining is None:
        remaining = quota = DEFAULT_CREDIT
        await client.hmset(user, {"remaining": str(remaining), "quota": str(DEFAULT_CREDIT)})
    return UsageData(quota=int(quota), remaining=int(remaining))

async def verify_cost(user:str, cost:int):
    if user is None:
        raise credentials_exception
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

async def ratelimit(ip:str, expensive:bool=False):
    ratelimit = await get_anon_ratelimit(expensive)
    res = await ratelimit.limit(ip)
    if not res.allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached! Please sign up for an account to continue. Free usage units will be granted on sign-up."
        )