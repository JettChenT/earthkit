"use client";
import { Coords } from "@/lib/geo";
import { useComb } from "@/lib/combStore";
import { columnHelper } from "../comb/table";
import { TableItemsFromCoord } from "@/lib/utils";

export interface ToCombOptions {
  coords: Coords;
  target_img: string;
}

export const pushToComb = ({ coords, target_img }: ToCombOptions) => {
  const { setColDef, addItems, setTargetImage } = useComb();
  setTargetImage(target_img);
  setColDef((colDefs) => [
    ...colDefs,
    columnHelper.accessor("aux.max_sim", {
      header: "StreetView Similarity",
      cell: ({ row }) => <div>{row.original.aux?.max_sim}</div>,
    }),
  ]);
  addItems(TableItemsFromCoord(coords));
};
