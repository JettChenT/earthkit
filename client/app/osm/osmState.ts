import { create } from "zustand";
import { Model } from "./actions";

export type OsmGlobs = {
  model: Model;
  setModel: (model: Model) => void;
  latestGeneration: string | null;
  setLatestGeneration: (generation_id: string) => void;
};

export const useOsmGlobs = create<OsmGlobs>((set) => ({
  model: "gpt-3.5-turbo",
  setModel: (model: Model) => set({ model }),
  latestGeneration: null,
  setLatestGeneration: (generation_id: string) =>
    set({ latestGeneration: generation_id }),
}));
