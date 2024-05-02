import { Coords } from "./geo";

export type BaseLayer = {
  type: "base";
  id: string;
};

export type ProbScatterLayer = {
  type: "prob_scatter";
  coords: Coords;
  key: string | null;
  id: string;
};

export type Layer = BaseLayer | ProbScatterLayer;
