from .otel import tracer_provider, instrument
from .cfig import ENVS
from .modal_otel import ModalInstrumentor, OTEL_DEPS
instrument()
import modal
import time

app = modal.App("tst")

image = (modal.Image
         .debian_slim(python_version="3.11")
         .pip_install(*OTEL_DEPS)
         .env(ENVS)
)

@app.function(image=image)
def sq(x):
    time.sleep(1)
    return x**2

@app.function(image=image)
def sq_sum():
    print('executing...')
    time.sleep(1)
    squares = sq.map(range(10))
    res = sum(squares)
    return res


@app.local_entrypoint()
async def main():
    import dotenv
    dotenv.load_dotenv()
    res = sq_sum.remote()
    print(res)

