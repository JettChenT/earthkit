import { Point } from "@/lib/geo";
import { OnChangeFn, SortingState, Updater } from "@tanstack/react-table";
import { create } from "zustand";

export type TableItem = {
  coord: Point;
  panoId?: string;
  status: LabelType;
};

export type ViewPanelType = "streetview" | "map" | "satellite";
export type LabelType = "Match" | "Keep" | "Not Match" | "Not Labeled";

export type CombState = {
  target_image: string | null;
  items: TableItem[];
  idx: number;
  viewPanelState: ViewPanelType;
  sorting: SortingState;
  setTargetImage: (img: string) => void;
  setItems: (items: TableItem[]) => void;
  setViewPanelState: (state: ViewPanelType) => void;
  getSelected: () => TableItem | null;
  idxDelta: (delta: number) => void;
  setIdx: (index: number) => void;
  setIdxData: (newItem: Partial<TableItem>) => void;
  setSorting: OnChangeFn<SortingState>;
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
      status: "Not Labeled",
    },
    {
      coord: {
        lat: 35.6588,
        lon: 139.4098,
        aux: {},
      },
      panoId: "hVQGOqoZekuaidl-60eDfA",
      status: "Not Labeled",
    },
  ],
  idx: 0,
  viewPanelState: "streetview",
  sorting: [
    {
      id: "status",
      desc: false,
    },
  ],
  setTargetImage: (img: string) => set(() => ({ target_image: img })),
  setItems: (items: TableItem[]) => set(() => ({ items })),
  setViewPanelState: (state: ViewPanelType) =>
    set(() => ({ viewPanelState: state })),
  getSelected: () => get().items[get().idx],
  idxDelta: (delta: number) =>
    set((state) => ({
      idx: Math.max(0, Math.min(state.items.length - 1, state.idx + delta)),
    })),
  setIdx: (index: number) =>
    set((state) => ({
      idx: Math.max(0, Math.min(state.items.length - 1, index)),
    })),
  setIdxData: (newItem: Partial<TableItem>) =>
    set((state) => {
      const updatedItems = [...state.items];
      updatedItems[state.idx] = { ...updatedItems[state.idx], ...newItem };
      return { items: updatedItems };
    }),
  setSorting: (fn) =>
    set((state) => ({
      sorting: fn instanceof Function ? fn(state.sorting) : fn,
    })),
}));
