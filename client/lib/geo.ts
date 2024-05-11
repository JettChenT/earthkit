export type Point = {
  lon: number;
  lat: number;
  aux: any;
};

export type Coords = {
  coords: Point[];
};

export type Bounds = {
  lo: Point;
  hi: Point;
};

export function getbbox(coords: Coords) {
  const lo = {
    lon: Math.min(...coords.coords.map((c) => c.lon)),
    lat: Math.min(...coords.coords.map((c) => c.lat)),
  };
  const hi = {
    lon: Math.max(...coords.coords.map((c) => c.lon)),
    lat: Math.max(...coords.coords.map((c) => c.lat)),
  };
  return { lo, hi };
}
