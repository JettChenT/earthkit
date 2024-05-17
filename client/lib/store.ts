import { create } from "zustand";
import { Resource } from "./resource";
import { Layer } from "./layers";

export type Tool = "geoclip" | "satellite" | "streetview";

export type AppState = {
  resources: Resource[];
  layers: Layer[];
  setResources: (resources: Resource[]) => void;
  setLayers: (layers: Layer[]) => void;
};

export const useStore = create<AppState>((set) => ({
  resources: [],
  layers: [],
  setResources: (resources: Resource[]) => set(() => ({ resources })),
  setLayers: (layers: Layer[]) => set(() => ({ layers })),
}));
