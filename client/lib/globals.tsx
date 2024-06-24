"use client";
import { create, useStore } from "zustand";

export type EKGlobals = {
  debug: boolean;
  setDebug: (debug: boolean) => void;
};

export const useEKGlobals = create<EKGlobals>((set) => ({
  debug: false,
  setDebug: (debug) => set({ debug }),
}));
