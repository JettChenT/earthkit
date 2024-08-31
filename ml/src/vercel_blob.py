# https://github.com/misaelnieto/vercel-storage/blob/main/vercel_storage/blob.py

from os import getenv
from typing import Any, Optional, Union
from mimetypes import guess_type
import urllib.parse
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

VERCEL_API_URL = "https://blob.vercel-storage.com"
TOKEN_ENV = "BLOB_READ_WRITE_TOKEN"
API_VERSION = "4"
DEFAULT_CACHE_AGE = 7 * 24 * 60 * 60  # 1 Week
DEFAULT_ACCESS = "public"
DEFAULT_PAGE_SIZE = 100


class ConfigurationError(ValueError):
    pass


class APIResponseError(ValueError):
    pass


def guess_mime_type(url):
    return guess_type(url, strict=False)[0]


def get_token(options: dict):
    _tkn = options.get("token", getenv(TOKEN_ENV, None))
    if not _tkn:
        raise ConfigurationError("Vercel's BLOB_READ_WRITE_TOKEN is not set")
    return _tkn


def dump_headers(options: Optional[dict], headers: dict):
    if options is None:
        options = {}


def _coerce_bool(value):
    return str(int(bool(value)))


async def _handle_response(response: httpx.Response):
    if str(response.status_code) == "200":
        return response.json()
    raise APIResponseError(f"Oops, something went wrong: {response.json()}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
async def put(
    pathname: str,
    body: bytes,
    mime: Optional[str] = None,
    options: Optional[dict] = None,
) -> dict:
    _opts = dict(options) if options else dict()
    headers = {
        "access": "public",
        "authorization": f"Bearer {get_token(_opts)}",
        "x-api-version": API_VERSION,
        "x-content-type": mime or guess_mime_type(pathname),
        "x-cache-control-max-age": _opts.get(
            "cacheControlMaxAge", str(DEFAULT_CACHE_AGE)
        ),
    }
    if options and "no_suffix" in options:
        headers["x-add-random-suffix"] = "false"

    dump_headers(options, headers)
    async with httpx.AsyncClient() as client:
        _resp = await client.put(
            f"{VERCEL_API_URL}/{pathname}", data=body, headers=headers, timeout=10.0
        )
    return await _handle_response(_resp)


async def delete(
    url: Union[str, list[str], tuple[str]], options: Optional[dict] = None
) -> dict:
    """
    Deletes a blob object from the Blob store.
    Args:
        url (str|list[str]|tuple[str]): A string or a list of strings specifying the
            unique URL(s) of the blob object(s) to delete.
        options (dict): A dict with the following optional parameter:
            token (Not required) A string specifying the read-write token to
                  use when making requests. It defaults to the BLOB_READ_WRITE_TOKEN
                  environment variable when deployed on Vercel as explained
                  in Read-write token

    Returns:
        None: A delete action is always successful if the blob url exists.
              A delete action won't throw if the blob url doesn't exists.
    """
    _opts = dict(options) if options else dict()
    headers = {
        "authorization": f"Bearer {get_token(_opts)}",
        "x-api-version": API_VERSION,
        "content-type": "application/json",
    }
    dump_headers(options, headers)
    async with httpx.AsyncClient() as client:
        _resp = await client.post(
            f"{VERCEL_API_URL}/delete",
            json={
                "urls": [
                    url,
                ]
                if isinstance(url, str)
                else url
            },
            headers=headers,
        )
    return await _handle_response(_resp)


async def list(options: Optional[dict] = None) -> Any:
    """
    The list method returns a list of blob objects in a Blob store.
    Args:
        options (dict): A dict with the following optional parameter:
            token (Not required) A string specifying the read-write token to
                  use when making requests. It defaults to the BLOB_READ_WRITE_TOKEN
                  environment variable when deployed on Vercel as explained
                  in Read-write token
            limit (Not required): A number specifying the maximum number of
                blob objects to return. It defaults to 1000
            prefix (Not required): A string used to filter for blob objects
                contained in a specific folder assuming that the folder name was
                used in the pathname when the blob object was uploaded
            cursor (Not required): A string obtained from a previous response for pagination
                of retults
            mode (Not required): A string specifying the response format. Can
                either be "expanded" (default) or "folded". In folded mode
                all blobs that are located inside a folder will be folded into
                a single folder string entry

    Returns:
        Json response with the following format:

        blobs: A list of blobs
        cursor: (Optional) You get this if you are doing pagination
        hasMore: boolean
        folders: A list of strings.
    """
    _opts = dict(options) if options else dict()
    headers = {
        "authorization": f"Bearer {get_token(_opts)}",
        "limit": _opts.get("limit", str(DEFAULT_PAGE_SIZE)),
    }
    if "prefix" in _opts:
        headers["prefix"] = _opts["prefix"]
    if "cursor" in _opts:
        headers["cursor"] = _opts["cursor"]
    if "mode" in _opts:
        headers["mode"] = _opts["mode"]

    dump_headers(options, headers)
    async with httpx.AsyncClient() as client:
        _resp = await client.get(
            f"{VERCEL_API_URL}",
            headers=headers,
        )
    return await _handle_response(_resp)


async def head(url: str, options: Optional[dict] = None) -> dict:
    """
    Returns a blob object's metadata.

    Args:
        url: (Required) A string specifying the unique URL of the blob object to read
        options (dict): A dict with the following optional parameter:
            token (Not required) A string specifying the read-write token to
                  use when making requests. It defaults to the BLOB_READ_WRITE_TOKEN
                  environment variable when deployed on Vercel as explained
                  in Read-write token

    Returns:
        dict: with the blob's metadata. Throws an error if the blob is not found
    """
    _opts = dict(options) if options else dict()
    headers = {
        "authorization": f"Bearer {get_token(_opts)}",
        "x-api-version": API_VERSION,
    }
    dump_headers(options, headers)
    async with httpx.AsyncClient() as client:
        _resp = await client.get(
            f"{VERCEL_API_URL}", headers=headers, params={"url": url}
        )
    return await _handle_response(_resp)


async def copy(from_url: str, to_pathname: str, options: Optional[dict] = None) -> dict:
    """
    Copies an existing blob object to a new path inside the blob store.

    The contentType and cacheControlMaxAge will not be copied from the source
    blob. If the values should be carried over to the copy, they need to be
    defined again in the options object.

    Contrary to put(), addRandomSuffix is false by default. This means no
    automatic random id suffix is added to your blob url, unless you pass
    addRandomSuffix: True. This also means copy() overwrites files per default,
    if the operation targets a pathname that already exists.

    Args:
        from_url: (Required) A blob URL identifying an already existing blob
        to_pathname: (Required) A string specifying the new path inside the blob
            store. This will be the base value of the return URL
        options: A dict with the following optional parameter:
            token (Not required): A string specifying the read-write token to
                  use when making requests. It defaults to the BLOB_READ_WRITE_TOKEN
                  environment variable when deployed on Vercel as explained
                  in Read-write token
            contentType (Not required): A string indicating the media type.
                By default, it's extracted from the to_pathname's extension.
            addRandomSuffix (Not required): A boolean specifying whether to add
                a random suffix to the pathname. It defaults to False.
            cacheControlMaxAge (Not required): A number in seconds to configure
                the edge and browser cache. Defaults to one year. See Vercel's
                caching documentation for more details.
    """
    _opts = dict(options) if options else dict()
    headers = {
        "access": "public",
        "authorization": f"Bearer {get_token(_opts)}",
        "x-api-version": API_VERSION,
        "x-content-type": _opts.get("contentType", guess_mime_type(from_url)),
        "x-add-random-suffix": _coerce_bool(_opts.get("addRandomSuffix", False)),
        "x-cache-control-max-age": _opts.get(
            "cacheControlMaxAge", str(DEFAULT_CACHE_AGE)
        ),
    }
    dump_headers(options, headers)
    _to = urllib.parse.quote(to_pathname)
    async with httpx.AsyncClient() as client:
        resp = await client.put(
            f"{VERCEL_API_URL}/{_to}", headers=headers, params={"fromUrl": from_url}
        )
    return await _handle_response(resp)


async def _main():
    from dotenv import load_dotenv

    load_dotenv()
    with open("img/isld.jpg", "rb") as f:
        resp = await put("tst/island.jpg", f.read())
    print(resp)


if __name__ == "__main__":
    asyncio.run(_main())
