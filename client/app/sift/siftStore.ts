import { Coords, PointFromPurePoint, PurePoint } from "@/lib/geo";
import {
  ColumnDef,
  ColumnFiltersState,
  OnChangeFn,
  SortingState,
} from "@tanstack/react-table";
import { create } from "zustand";
import { MOCK, mockCols, mockItems } from "./mock";
import { Col, defaultCols, mergeCols } from "@/app/sift/cols";
import { TableEncapsulation } from "./inout";
import { ResultsUpdate } from "@/lib/rpc";

export type TableItem = {
  coord: PurePoint;
  status: LabelType;
  aux: any;
};

export type ViewPanelType = "overview" | "streetview" | "map" | "satellite";
export type LabelType = "Match" | "Keep" | "Not Match" | "Not Labeled";
export const FiltPresets = {
  Match: ["Match"],
  Keep: ["Keep"],
  NotMatch: ["Not Match"],
  NotLabeled: ["Not Labeled"],
  Labeled: ["Match", "Keep", "Not Match"],
  All: ["Match", "Keep", "Not Match", "Not Labeled"],
};

export type SiftState = {
  target_image: string | null;
  items: TableItem[];
  idx: number;
  viewPanelState: ViewPanelType;
  sorting: SortingState;
  filtering: ColumnFiltersState;
  cols: Col[];
  updateItemResults: (res: ResultsUpdate, feat: string) => void;
  setTargetImage: (img: string | null) => void;
  setItems: (items: TableItem[]) => void;
  addItems: (items: TableItem[]) => void;
  tableImport: (tabl: TableEncapsulation) => void;
  setViewPanelState: (state: ViewPanelType) => void;
  getSelected: () => TableItem | null;
  getCoords: () => Coords;
  idxDelta: (delta: number) => void;
  setIdx: (index: number) => void;
  setIdxData: (newItem: Partial<TableItem>) => void;
  setSorting: OnChangeFn<SortingState>;
  setFiltering: OnChangeFn<ColumnFiltersState>;
  setCols: OnChangeFn<Col[]>;
  clearTable: () => void;
};

export const useSift = create<SiftState>((set, get) => ({
  target_image: null,
  items: MOCK ? mockItems : [],
  idx: 0,
  viewPanelState: "overview",
  cols: MOCK ? mockCols : defaultCols,
  sorting: [],
  filtering: [
    {
      id: "status",
      value: FiltPresets.All,
    },
  ],
  updateItemResults: (res: ResultsUpdate, feat: string) =>
    set((state) => ({
      items: state.items.map((item, idx) => {
        const result = res.results.find((result) => result.idx === idx);
        if (result) {
          return {
            ...item,
            aux: {
              ...item.aux,
              [feat]: result.value,
            },
          };
        }
        return item;
      }),
    })),
  setTargetImage: (img: string | null) => set(() => ({ target_image: img })),
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
  setCols: (fn) =>
    set((state) => ({
      cols: fn instanceof Function ? fn(state.cols) : fn,
    })),
  tableImport: (table: TableEncapsulation) =>
    set((state) => ({
      items: [...state.items, ...table.items],
      cols: mergeCols(state.cols, table.cols),
    })),
  getCoords: () => ({
    coords: get().items.map((item) => PointFromPurePoint(item.coord, {})),
  }),
  clearTable: () =>
    set(() => ({
      items: [],
      idx: 0,
      cols: defaultCols,
      sorting: [],
      filtering: [],
      viewPanelState: "overview",
    })),
}));
