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
    time.sleep(1)
    print("done")
    return sq(x)

@app.function(image=image)
def sq_iter(n):
    for x in range(n):
        yield tst_fnc.remote(x)

@app.function(image=image)
def sq_sum():
    print('executing...')
    time.sleep(1)
    s = 0
    for res in sq_iter.remote_gen(10):
        s += res
    return s


@app.local_entrypoint()
async def main():
    res = sq_sum.remote()
    print(res)

