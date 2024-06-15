import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { TableItem } from "./combStore";
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
  if (value === null) {
    return "N/A";
  }
  if (typeof value === "number" && !Number.isInteger(value)) {
    return value.toFixed(3);
  }
  return value;
}
