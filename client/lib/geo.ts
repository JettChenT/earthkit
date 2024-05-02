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
