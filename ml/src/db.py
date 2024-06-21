import os
from supabase import AClient
from attrs import define
from fastapi import HTTPException, status
import cattrs

def get_client():
    SUPABASE_URL = os.getenv("SUPABASE_URL") or ""
    SUPABASE_KEY = os.getenv("SUPABASE_KEY") or ""
    return AClient(SUPABASE_URL, SUPABASE_KEY)

@define
class Quota:
    id: str
    quota: int
    used: int
    renew_month: int

async def get_quota(user:str):
    response = (get_client()
                .table('usage')
                .select('*')
                .eq('id', user)
                .single())
    data = await response.execute()
    return cattrs.structure(data.data, Quota)


async def verify_cost(user:str, cost:int):
    quota = await get_quota(user)
    if quota.used + cost > quota.quota:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail=f"Quota exceeded. This operation requires {cost} tokens, you have {quota.quota - quota.used} tokens left."
                    "Please contact contact@earthkit.app for a higher quota."
        )
    
    upd = (get_client()
            .table('usage')
            .update({'used': quota.used + cost})
            .eq('id', user))
    return await upd.execute()
