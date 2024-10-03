import modal
from modal import Image, App, gpu, build, enter, method
import argparse
import matplotlib.pyplot as plt
import numpy as np
from .maploc.demo import Demo
from .maploc.osm.tiling import TileManager
from .maploc.osm.viz import Colormap
from .geo import Point
from concurrent.futures import ThreadPoolExecutor
import time


app = App("orienternet")

image = (Image.debian_slim(python_version="3.11")
         .apt_install("git","libgl1-mesa-glx","libglib2.0-0")
         .pip_install_from_pyproject("pyproject.toml"))

NUM_ROTATIONS = 256

@app.cls(image=image, gpu=gpu.A100(), enable_memory_snapshot=True)
class OrienterNetModel:
    @build()
    def build(self):
        _ = Demo(num_rotations=NUM_ROTATIONS)

    @enter(snap=True)
    def load(self):
        self.demo = Demo(num_rotations=NUM_ROTATIONS)
    
    @enter(snap=False)
    def setup(self):
        self.demo.to("cuda")

    @method()
    def locate(self, image_path: str, prior: Point, tile_size: int = 128):
        start_time = time.time()

        # Read input image and prepare data
        def process_image():
            return self.demo.read_input_image(
                image_path,
                prior_latlon=(prior.lat, prior.lon),
                tile_size_meters=tile_size,
            )

        def get_canvas():
            proj, bbox = self.demo.proc_prior((prior.lat, prior.lon), tile_size)
            tiler = TileManager.from_bbox(proj, bbox + 10, self.demo.config.data.pixel_per_meter)
            return proj, bbox, tiler.query(bbox)

        with ThreadPoolExecutor() as executor:
            # Submit the image processing task
            image_task = executor.submit(process_image)

            # Submit the canvas retrieval task
            canvas_task = executor.submit(get_canvas)

            # Get results from both tasks
            image, camera, gravity = image_task.result()
            proj, bbox, canvas = canvas_task.result()

        parallel_time = time.time() - start_time
        print(f"Parallel processing time: {parallel_time:.2f} seconds")

        # Run the inference
        inference_start = time.time()
        uv = self.demo.localize(
            image, camera, canvas, gravity=gravity
        )
        inference_time = time.time() - inference_start
        print(f"Inference time: {inference_time:.2f} seconds")

        # Convert UV coordinates to lat/lon
        conversion_start = time.time()
        calibrated_xy = canvas.to_xy(uv)
        calibrated_latlon = proj.unproject(calibrated_xy)
        conversion_time = time.time() - conversion_start
        print(f"Coordinate conversion time: {conversion_time:.2f} seconds")

        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")

        return Point(lat=calibrated_latlon[0], lon=calibrated_latlon[1])
@app.local_entrypoint()
def main():
    import time
    # Download the image from the provided URL
    image_url = "https://jld59we6hmprlla0.public.blob.vercel-storage.com/earthkit_uploads/query_zurich_1-VDYr4Uaxlus9SpvTdvubPnMYEJ1iWh.JPG"

    lat = 47.37849417235291
    lon = 8.548809525913553
    tile_size = 128

    start = time.time()
    result = OrienterNetModel().locate.remote(image_url, Point(lat=lat, lon=lon), tile_size)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    
    print(f"Input coordinates: {lat}, {lon}")
    print(f"Calibrated coordinates: {result.lat:.6f}, {result.lon:.6f}")
