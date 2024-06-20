import asyncio

from aiostream import pipe, stream


def mlt(a:int, b:int) -> int:
    print(a,b)
    return a*b


async def main() -> None:
    xs = stream.count(interval=0.1)
    ys = xs | pipe.enumerate() | pipe.starmap(mlt)

    async with ys.stream() as streamer:
        async for y in streamer:
            print(y)

# Run main coroutine
asyncio.run(main())