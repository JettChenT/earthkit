from typing import Optional
from fastapi import HTTPException, Request, status, Depends
import jwt
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel
import base64
import os

PUB_KEY = base64.b64decode(os.getenv("JWT_PK") or "")
ALGORITHM = "RS256"

oauth2_scheme = OAuth2AuthorizationCodeBearer(authorizationUrl="foo", tokenUrl="token", auto_error=False)

class Token(BaseModel):
    access_token: str
    token_type: str

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid authentication credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, PUB_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except jwt.PyJWTError as e:
        print(e)
        raise credentials_exception


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme, use_cache=False)):
    if token is None:
        return None
    return decode_access_token(token)
