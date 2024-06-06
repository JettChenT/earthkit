"use client";
import { LabelType, TableItem, useComb } from "./combStore";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  createColumnHelper,
  getSortedRowModel,
} from "@tanstack/react-table";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import Pill, { PillColor } from "@/components/pill";
import { useHotkeys } from "react-hotkeys-hook";
import { Key } from "ts-key-enum";
import { useEffect, useState } from "react";

const columnHelper = createColumnHelper<TableItem>();

const columnsBase = [
  columnHelper.accessor("coord", {
    header: "Coordinate",
    cell: (props) => {
      const coord = props.getValue();
      return (
        <Pill color="blue">
          {coord.lat}, {coord.lon}
        </Pill>
      );
    },
  }),
  columnHelper.accessor("status", {
    header: "Status",
    cell: (props) => {
      const status = props.getValue();
      return <StatusCell status={status} />;
    },
    sortingFn: (a, b, cid) => {
      return (
        StatusNumberMap[a.original.status] - StatusNumberMap[b.original.status]
      );
    },
  }),
  columnHelper.accessor("panoId", {
    header: "Pano ID",
  }),
];

const StatusToPillColor = (status: LabelType): PillColor => {
  switch (status) {
    case "Not Labeled":
      return "grey";
    case "Match":
      return "green";
    case "Keep":
      return "blue";
    case "Not Match":
      return "red";
  }
};

const StatusNumberMap: Record<LabelType, number> = {
  "Not Labeled": 0,
  Match: 1,
  Keep: 2,
  "Not Match": 3,
};

function StatusCell({ status }: { status: LabelType }) {
  const col: PillColor = StatusToPillColor(status);
  return <Pill color={col}>{status}</Pill>;
}

export default function CombTable() {
  let { items, idx, setIdx, sorting, setSorting } = useComb();
  let [selectedIdx, setSelectedIdx] = useState(0);

  const table = useReactTable({
    data: items,
    columns: columnsBase,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    state: {
      sorting,
    },
  });

  useEffect(() => {
    const actualIdx = table.getRowModel().rows.at(selectedIdx)!.index;
    setIdx(actualIdx);
  }, [items, selectedIdx]);

  const idxDelta = (delta: number) => {
    setSelectedIdx((prev) =>
      Math.max(0, Math.min(prev + delta, items.length - 1))
    );
  };

  useHotkeys(["j", Key.ArrowDown], () => {
    idxDelta(1);
  });
  useHotkeys(["k", Key.ArrowUp], () => {
    idxDelta(-1);
  });

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => {
                return (
                  <TableHead key={header.id}>
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                  </TableHead>
                );
              })}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row, dispIdx) => (
              <TableRow
                key={row.id}
                data-state={
                  (row.getIsSelected() || row.index === idx) && "selected"
                }
                onMouseDown={() => {
                  setSelectedIdx(dispIdx);
                }}
                className="cursor-pointer"
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell
                colSpan={columnsBase.length}
                className="h-24 text-center"
              >
                No results.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
