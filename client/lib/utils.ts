import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { TableItem } from "../app/sift/siftStore";
import { Coords, Point, PurePointFromPoint } from "./geo";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function TableItemFromPoint(pnt: Point): TableItem {
  return {
    coord: PurePointFromPoint(pnt),
    status: "Not Labeled",
    aux: pnt.aux,
  };
}

export function TableItemsFromCoord(coord: Coords): TableItem[] {
  return coord.coords.map(TableItemFromPoint);
}

export function formatValue(value: any) {
  if (typeof value === "number" && !Number.isInteger(value)) {
    return value.toFixed(3);
  }
  return value;
}

export type Stats = {
  mean: number;
  stdev: number;
};

export function getStats(values: number[]): Stats {
  const mean = values.reduce((acc, value) => acc + value, 0) / values.length;
  const variance =
    values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / values.length;
  return {
    mean,
    stdev: Math.sqrt(variance),
  };
}

export function zVal(value: number, stats: Stats) {
  return (value - stats.mean) / stats.stdev;
}

export const downloadContent = (content: string, ext: string) => {
  const a = document.createElement("a");
  const file = new Blob([content], { type: `text/${ext}` });
  a.href = URL.createObjectURL(file);
  a.download = `export.${ext}`;
  a.click();
};
