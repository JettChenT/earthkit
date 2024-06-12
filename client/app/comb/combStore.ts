import { Point, PurePoint } from "@/lib/geo";
import {
  ColumnDef,
  ColumnFiltersState,
  OnChangeFn,
  SortingState,
} from "@tanstack/react-table";
import { create } from "zustand";
import { MOCK, mockItems } from "./mock";
import { columnsBase } from "./table";

export type TableItem = {
  coord: PurePoint;
  panoId?: string;
  status: LabelType;
};

export type ViewPanelType = "streetview" | "map" | "satellite";
export type LabelType = "Match" | "Keep" | "Not Match" | "Not Labeled";
export const FiltPresets = {
  Match: ["Match"],
  Keep: ["Keep"],
  NotMatch: ["Not Match"],
  NotLabeled: ["Not Labeled"],
  Labeled: ["Match", "Keep", "Not Match"],
  All: ["Match", "Keep", "Not Match", "Not Labeled"],
};

export type CombState = {
  target_image: string | null;
  items: TableItem[];
  idx: number;
  viewPanelState: ViewPanelType;
  sorting: SortingState;
  filtering: ColumnFiltersState;
  colDefs: ColumnDef<TableItem, any>[];
  setTargetImage: (img: string) => void;
  setItems: (items: TableItem[]) => void;
  addItems: (items: TableItem[]) => void;
  setViewPanelState: (state: ViewPanelType) => void;
  getSelected: () => TableItem | null;
  idxDelta: (delta: number) => void;
  setIdx: (index: number) => void;
  setIdxData: (newItem: Partial<TableItem>) => void;
  setSorting: OnChangeFn<SortingState>;
  setFiltering: OnChangeFn<ColumnFiltersState>;
};

export const useComb = create<CombState>((set, get) => ({
  target_image: null,
  items: MOCK ? mockItems : [],
  idx: 0,
  viewPanelState: "streetview",
  sorting: [
    {
      id: "status",
      desc: false,
    },
  ],
  filtering: [
    {
      id: "status",
      value: FiltPresets.All,
    },
  ],
  colDefs: columnsBase,
  setTargetImage: (img: string) => set(() => ({ target_image: img })),
  setItems: (items: TableItem[]) => set(() => ({ items })),
  addItems: (newItems: TableItem[]) =>
    set((state) => ({ items: [...state.items, ...newItems] })),
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
  setFiltering: (fn) =>
    set((state) => ({
      filtering: fn instanceof Function ? fn(state.filtering) : fn,
    })),
}));
