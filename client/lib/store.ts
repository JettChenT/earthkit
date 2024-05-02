import { create } from "zustand";
import { Resource } from "./resource";
import { Layer } from "./layers";

export type Tool = "geoclip" | "satellite" | "streetview";

export type AppState = {
  resources: Resource[];
  layers: Layer[];
  tool: Tool;
  setResources: (resources: Resource[]) => void;
  setLayers: (layers: Layer[]) => void;
  setTool: (tool: Tool) => void;
};

export const useStore = create<AppState>((set) => ({
  resources: [],
  layers: [],
  tool: "geoclip",
  setResources: (resources: Resource[]) => set(() => ({ resources })),
  setLayers: (layers: Layer[]) => set(() => ({ layers })),
  setTool: (tool: Tool) => set(() => ({ tool })),
}));
