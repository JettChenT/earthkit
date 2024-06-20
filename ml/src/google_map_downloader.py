from PIL import Image   
import urllib.request
import os
import math
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import List
from tqdm import tqdm
from httpx import AsyncClient
from .geo import Point, Bounds, Coords

def lat_lng_to_tile(point: Point, zoom):
    tile_size = 256 
    numTiles = 1 << zoom
    point_x = (tile_size / 2 + point.lon * tile_size / 360.0) * numTiles // tile_size
    sin_y = math.sin(point.lat * (math.pi / 180.0))
    point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(tile_size / (2 * math.pi))) * numTiles // tile_size
    return int(point_x), int(point_y)

def tile_to_lat_lng(x, y, zoom) -> Point:
    numTiles = 1 << zoom
    lon = -180.0 + 360.0 * x / numTiles
    s_y = math.tanh(math.pi * (numTiles - 2.0 * y) / numTiles)   
    lat = math.degrees(math.asin(s_y))
    return Point(lon=lon, lat=lat)


def download_tile(x, y, zoom):
    url = f'https://mt0.google.com/vt/lyrs=s&?x={x}&y={y}&z={zoom}'
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
        image = BytesIO(image_data).read()
    pnt = tile_to_lat_lng(x, y, zoom)
    # print(f"Downloading tile: {x}, {y}, {zoom}")
    pnt.aux = {'image': image, 'x': x, 'y': y, 'zoom': zoom}
    return pnt

asclient = AsyncClient()

async def download_tile_async(x, y, zoom, lyr="s"):
    url = f'https://mt0.google.com/vt/lyrs={lyr}&?x={x}&y={y}&z={zoom}'
    resp = await asclient.get(url)
    resp.raise_for_status()
    image_data = resp.content
    image = Image.open(BytesIO(image_data))
    return image

async def download_point_async(pnt: Point, zoom=17, lyr="s"):
    x, y = lat_lng_to_tile(pnt, zoom)
    return await download_tile_async(x, y, zoom, lyr)

def download_google_map_area(bounds: Bounds, zoom=17) -> Coords:
    """
    Downloads Google map tiles given a bounding box and a zoom level.
    
    Args:
        bounds: Bounds object containing two Points (bottom-left and top-right).
        zoom: Zoom level for the tiles.
    
    Returns:
        A list of tuples, each containing the tile coordinates and the tile image.
    """
    l_x, l_y = lat_lng_to_tile(bounds.lo, zoom)
    r_x, r_y = lat_lng_to_tile(bounds.hi, zoom)
    start_x, start_y = min(l_x, r_x), min(l_y, r_y)
    end_x, end_y = max(l_x, r_x), max(l_y, r_y)
    print(f"Starting tile: {start_x}, {start_y}")
    print(f"Ending tile: {end_x}, {end_y}")

    with ThreadPoolExecutor() as executor:
        total_tiles = (end_x - start_x + 1) * (end_y - start_y + 1)
        print(f"Downloading {total_tiles} tiles")
        futures = [executor.submit(download_tile, x, y, zoom) for x in range(start_x, end_x + 1) for y in range(start_y, end_y + 1)]
        results = []
        for future in tqdm(futures, total=total_tiles, desc="Downloading tiles"):
            results.append(future.result())

    return Coords(results)

def download_sat_coords(coords: Coords, zoom=17) -> Coords:
    """
    Downloads Google map tiles given a list of coordinates and a zoom level.
    
    Args:
        coords: Coords object containing a list of Points.
        zoom: Zoom level for the tiles.
    
    Returns:
        A list of tuples, each containing the tile coordinates and the tile image.
    """
    with ThreadPoolExecutor() as executor:
        total_tiles = len(coords)
        print(f"Downloading {total_tiles} tiles")
        futures = [executor.submit(download_tile, *lat_lng_to_tile(point, zoom), zoom) for point in coords]
        results = []
        for future in tqdm(futures, total=total_tiles, desc="Downloading tiles"):
            results.append(future.result())

    return Coords(results)

def tst_math():
    import random
    for _ in range(10):
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        point = Point(lon=lon, lat=lat)
        zm = 20
        x, y = lat_lng_to_tile(point, zm)
        rev_pnt = tile_to_lat_lng(x, y, zm)
        print(f"Lat: {lat}, Lon: {lon}, X: {x}, Y: {y}\n Rev Lat: {rev_pnt.lat}, Rev Lon: {rev_pnt.lon}\n")

def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import os
    from PIL import Image

    bounds = Bounds.from_points(
        Point(lat=37.789733, lon=-122.402614), Point(lat=37.784409, lon=-122.394974)
    )
    zoom = 20 
    try:
        tiles = download_google_map_area(bounds, zoom)
        print(f"{len(tiles)} tiles downloaded")
        if not os.path.exists('tiles'):
            os.makedirs('tiles')
        tile_width, tile_height = Image.open(BytesIO(tiles[0].aux['image'])).size
        total_width = (max(x for x, y, image in [(tile.aux['x'], tile.aux['y'], tile.aux['image']) for tile in tiles]) - min(x for x, y, image in [(tile.aux['x'], tile.aux['y'], tile.aux['image']) for tile in tiles]) + 1) * tile_width
        total_height = (max(y for x, y, image in [(tile.aux['x'], tile.aux['y'], tile.aux['image']) for tile in tiles]) - min(y for x, y, image in [(tile.aux['x'], tile.aux['y'], tile.aux['image']) for tile in tiles]) + 1) * tile_height
        stitched_image = Image.new('RGB', (total_width, total_height))
        fig, ax = plt.subplots()
        for tile in tiles:
            x, y, image = tile.aux['x'], tile.aux['y'], Image.open(BytesIO(tile.aux['image']))
            print(x,y,tile.lat,tile.lon)
            top_left_x = (x - min(x for x, y, image in [(tile.aux['x'], tile.aux['y'], tile.aux['image']) for tile in tiles])) * tile_width
            top_left_y = (y - min(y for x, y, image in [(tile.aux['x'], tile.aux['y'], tile.aux['image']) for tile in tiles])) * tile_height
            stitched_image.paste(image, (top_left_x, top_left_y))
            rect = Rectangle((top_left_x, top_left_y), tile_width, tile_height, fill=False, color='red')
            image.save(f"tiles/{x}_{y}_{zoom}.png")
            ax.add_patch(rect)

        ax.imshow(stitched_image)
        stitched_image.save("stitched_map.png")
        fig.savefig("boundaries.png")
        print("The map has successfully been created")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
