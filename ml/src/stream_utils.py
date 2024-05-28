from typing import AsyncIterable, AsyncIterator, TypeVar
from aiostream import stream, pipe, pipable_operator, streamcontext

T = TypeVar("T")

@pipable_operator
async def eager_chunks(source: AsyncIterable[T], n: int) -> AsyncIterator[list[T]]:
    enumerated = stream.enumerate(source)
    async with streamcontext(enumerated) as streamer:
        lst = []
        if n<=0:
            return
        async for i, item in streamer:
            lst.append(item)
            if (i+1) % n == 0: 
                assert len(lst) == n
                yield lst
                lst = []
        if lst:
            yield lst

async def main():
    async for chunk in eager_chunks(stream.iterate(range(10)), 3):
        print(chunk)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

