import modal
import urllib.request
from .geo import Point, Coords

stub = modal.Stub("tst")

image = modal.Image.debian_slim(python_version="3.11")


def sq(n: int):
    return n * n


@stub.function(image=image)
def tst_fnc():
    pnt = Point(1.1, 2.2)
    coords = Coords()
    coords.append(pnt)
    return coords


@stub.local_entrypoint()
def main():
    print(tst_fnc.remote())
