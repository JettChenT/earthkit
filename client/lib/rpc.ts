import { Coords } from "./geo";

export type CoordMsg = Coords & { type: "Coords" };
export type ProgressMsg = {
  type: "Progress";
  progress: number;
};

export type Msg = CoordMsg | ProgressMsg;
