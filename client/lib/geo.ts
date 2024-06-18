import { ResultsUpdate } from "./rpc";

export interface Point {
  lon: number;
  lat: number;
  aux: any;
}

export interface PurePoint {
  lon: number;
  lat: number;
}

export const PurePointFromPoint = ({ lon, lat }: Point): PurePoint => ({
  lon,
  lat,
});

export const PointFromPurePoint = (
  { lon, lat }: PurePoint,
  aux: any
): Point => ({
  lon,
  lat,
  aux: aux === undefined ? {} : aux,
});

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

export function applyResultsUpdate(
  original: Coords,
  update: ResultsUpdate,
  facet: string
): Coords {
  const updatedCoords = original.coords.map((point, idx) => {
    const result = update.results.find((result) => result.idx === idx);
    if (result) {
      return {
        ...point,
        aux: {
          ...point.aux,
          [facet]: result.value,
        },
      };
    }
    return point;
  });

  return { coords: updatedCoords };
}
