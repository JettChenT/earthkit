import { Point } from "@/lib/geo";
import { create } from "zustand";

export type TableItem = {
  coord: Point;
  panoId?: string;
};

export type ViewPanelType = "streetview" | "map" | "satellite";

export type CombState = {
  target_image: string | null;
  items: TableItem[];
  idx: number;
  viewPanelState: ViewPanelType;
  setTargetImage: (img: string) => void;
  setItems: (items: TableItem[]) => void;
  setViewPanelState: (state: ViewPanelType) => void;
  getSelected: () => TableItem | null;
  idxDelta: (delta: number) => void;
};

export const useComb = create<CombState>((set, get) => ({
  target_image: null,
  items: [
    {
      coord: {
        lat: 35.6587,
        lon: 139.4089,
        aux: {},
      },
      panoId: "VcX0mBaFgJXXzvsZ8uu9rA",
    },
    {
      coord: {
        lat: 35.6588,
        lon: 139.4098,
        aux: {},
      },
      panoId: "hVQGOqoZekuaidl-60eDfA",
    },
  ],
  idx: 0,
  viewPanelState: "streetview",
  setTargetImage: (img: string) => set(() => ({ target_image: img })),
  setItems: (items: TableItem[]) => set(() => ({ items })),
  setViewPanelState: (state: ViewPanelType) =>
    set(() => ({ viewPanelState: state })),
  getSelected: () => get().items[get().idx],
  idxDelta: (delta: number) =>
    set((state) => ({
      idx: Math.max(0, Math.min(state.items.length - 1, state.idx + delta)),
    })),
}));
