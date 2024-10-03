import { ResultsUpdate } from "./rpc";
import * as turf from "@turf/turf";

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

export type PureBounds = {
  lo: PurePoint;
  hi: PurePoint;
};

export function getbbox(coords: PurePoint[]) {
  const lo = {
    lon: Math.min(...coords.map((c) => c.lon)),
    lat: Math.min(...coords.map((c) => c.lat)),
  };
  const hi = {
    lon: Math.max(...coords.map((c) => c.lon)),
    lat: Math.max(...coords.map((c) => c.lat)),
  };
  return { lo, hi };
}

function pntToGeojson(from: PurePoint) {
  return [from.lon, from.lat];
}

function pntFromGeojson(from: [number, number]): Point {
  return { lon: from[0], lat: from[1], aux: {} };
}

export function getGridSample(
  bbox: PureBounds,
  sample_dist_km: number
): Point[] {
  const { lo, hi } = bbox;
  const turfBbox = [lo.lon, lo.lat, hi.lon, hi.lat] as [
    number,
    number,
    number,
    number
  ];
  const options = { units: "kilometers" as const };
  const hor_dist = turf.distance([lo.lon, lo.lat], [hi.lon, lo.lat], options);
  const vert_dist = turf.distance([lo.lon, lo.lat], [lo.lon, hi.lat], options);
  const items_cnt =
    Math.ceil(hor_dist / sample_dist_km) *
    Math.ceil(vert_dist / sample_dist_km);
  if (items_cnt > 100_000) {
    throw new Error("Too many samples!");
  }
  const result = turf.pointGrid(turfBbox, sample_dist_km, options);
  return result.features.map((feat) => {
    return pntFromGeojson(feat.geometry.coordinates as [number, number]);
  });
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

export function getBoundingBox(point: Point, tileSize: number): number[][] {
  const metersPerDegree = 111319.9;
  const latDelta = tileSize / metersPerDegree;
  const lonDelta =
    tileSize / (metersPerDegree * Math.cos((point.lat * Math.PI) / 180));

  return [
    [point.lon - lonDelta, point.lat - latDelta],
    [point.lon + lonDelta, point.lat - latDelta],
    [point.lon + lonDelta, point.lat + latDelta],
    [point.lon - lonDelta, point.lat + latDelta],
    [point.lon - lonDelta, point.lat - latDelta],
  ];
}
