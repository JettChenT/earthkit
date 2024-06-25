"use client";
import { create, useStore } from "zustand";

export type EKGlobals = {
  debug: boolean;
  setDebug: (debug: boolean) => void;
  sidebarExpanded: boolean;
  setSidebarExpanded: (sidebarExpanded: boolean) => void;
};

export const useEKGlobals = create<EKGlobals>((set) => ({
  debug: false,
  setDebug: (debug) => set({ debug }),
  sidebarExpanded: true,
  setSidebarExpanded: (sidebarExpanded) => set({ sidebarExpanded }),
}));
