import modal
import time

app = modal.App("tst")

image = (modal.Image
         .debian_slim(python_version="3.11")
         .pip_install_from_pyproject("pyproject.toml")
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

