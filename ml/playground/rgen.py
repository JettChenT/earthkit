import modal
from asyncio import sleep
app = modal.App("ek-playground-remotegen")

@app.function()
async def f(i, itvl):
    for j in range(i):
        await sleep(itvl)
        yield j


@app.local_entrypoint()
async def run_async():
    async for r in f.remote_gen.aio(10, 0.5):
        print(r)