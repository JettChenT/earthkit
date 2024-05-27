import modal
from .otel import tracer_provider
from .cfig import ENVS
import asyncio
from .modal_otel import ModalInstrumentor, OTEL_DEPS
import time

ModalInstrumentor().instrument(tracer_provider=tracer_provider)

app = modal.App("tst")

# ModalInstrumentor().instrument_app(app, tracer_provider=tracer_provider)

image = (modal.Image
         .debian_slim(python_version="3.11")
         .pip_install(*OTEL_DEPS)
         .env(ENVS)
)


def sq(n: int):
    return n * n

@app.function(image=image)
async def tst_fnc(x):
    print('executing...')
    await asyncio.sleep(2.5)
    print("done")
    return sq(x)

@app.function(image=image)
def tst_fnc2():
    print('executing...')
    time.sleep(2.5)
    res = tst_fnc.map([1,2,3,4,5,6,7])
    print("done")
    return res


@app.local_entrypoint()
async def main():
    res = tst_fnc2.remote()
    print(res)

