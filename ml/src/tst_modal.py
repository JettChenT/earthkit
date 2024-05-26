import modal
from .otel import tracer_provider
from .cfig import ENVS
from .modal_otel import ModalInstrumentor, OTEL_DEPS
import time


app = modal.App("tst")

ModalInstrumentor().instrument_app(app, tracer_provider=tracer_provider)

image = (modal.Image
         .debian_slim(python_version="3.11")
         .pip_install(*OTEL_DEPS)
         .env(ENVS)
)


def sq(n: int):
    return n * n

@app.function(image=image)
def tst_fnc():
    print('executing...')
    x = 12
    time.sleep(2.5)
    print("done")
    return sq(x)


@app.local_entrypoint()
async def main():
    res = tst_fnc.remote()
    print(res)

