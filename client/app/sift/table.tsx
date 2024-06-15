"use client";
import {
  FiltPresets,
  LabelType,
  TableItem,
  useSift,
} from "../../lib/siftStore";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  createColumnHelper,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
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
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import {
  FileDown,
  FileInput,
  FileJson2,
  Filter,
  Globe,
  Table as TableIcn,
} from "lucide-react";
import { Dialog, DialogTrigger } from "@/components/ui/dialog";
import { GeoImport } from "./geo-import";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { FormatType, exportData } from "./inout";
import { downloadContent, formatValue } from "@/lib/utils";

export const columnHelper = createColumnHelper<TableItem>();

export const columnsBase = [
  columnHelper.accessor("coord", {
    header: "Coordinate",
    cell: (props) => {
      const coord = props.getValue();
      return (
        <Pill color="blue">
          {formatValue(coord.lat)}, {formatValue(coord.lon)}
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
    filterFn: (row, columnFilter, filterValue) => {
      return filterValue.includes(row.getValue(columnFilter));
    },
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

// Note: order matters here
const sttDescriptionsj = {
  All: "All",
  Labeled: "Labeled",
  NotLabeled: "Not Labeled",
  Match: "Match",
  Keep: "Keep",
  NotMatch: "Not Match",
};

function StatusCell({ status }: { status: LabelType }) {
  const col: PillColor = StatusToPillColor(status);
  return <Pill color={col}>{status}</Pill>;
}

export default function SiftTable() {
  let {
    items,
    idx,
    setIdx,
    sorting,
    setSorting,
    filtering,
    setFiltering,
    colDefs,
  } = useSift();
  let [selectedIdx, setSelectedIdx] = useState(0);
  const tableContainerRef = useRef<HTMLDivElement | null>(null);

  const table = useReactTable({
    data: items,
    columns: colDefs,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onSortingChange: setSorting,
    onColumnFiltersChange: setFiltering,
    state: {
      sorting,
      columnFilters: filtering,
    },
  });

  useEffect(() => {
    const actualIdx = table.getRowModel().rows.at(selectedIdx)?.index ?? 0;
    setIdx(actualIdx);

    // Ensure the selected item is in view
    setTimeout(() => {
      const tableContainer = tableContainerRef.current;
      const selectedRow = tableContainer?.querySelector(
        `[data-state="selected"]`
      );
      if (selectedRow) {
        if (selectedRow.getBoundingClientRect().top < 0) {
          selectedRow.scrollIntoView({
            behavior: "auto",
            block: "start",
            inline: "nearest",
          });
        }
        if (selectedRow.getBoundingClientRect().bottom > window.innerHeight) {
          selectedRow.scrollIntoView({
            behavior: "auto",
            block: "end",
            inline: "nearest",
          });
        }
      }
    }, 10);
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
    <div className="flex flex-col gap-4 p-2 h-full">
      <div className="flex justify-between">
        <span className="text-md">{items.length} items</span>
        <div className="flex gap-2">
          <ImportBtn />
          <ExportBtn />
          <StatusFilterSelect />
        </div>
      </div>
      <div
        className="rounded-md border h-full overflow-y-auto"
        ref={tableContainerRef}
      >
        <Table className="h-full">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id} className="">
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
          <TableBody className="h-full">
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
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
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
    </div>
  );
}

function ImportBtn() {
  const [open, setOpen] = useState(false);
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline">
          <FileInput className="h-4 w-4 mr-1" />
          Import
        </Button>
      </DialogTrigger>
      <GeoImport setOpen={setOpen} />
    </Dialog>
  );
}

// TODO: perhaps look into https://www.npmjs.com/package/react-file-icon
const ExportFormats: { name: string; ext: FormatType; icon: JSX.Element }[] = [
  {
    name: "CSV",
    ext: "csv",
    icon: <TableIcn className="size-4 mr-3" />,
  },
  {
    name: "GeoJSON",
    ext: "geojson",
    icon: <Globe className="size-4 mr-3" />,
  },
  {
    name: "JSON",
    ext: "json",
    icon: <FileJson2 className="size-4 mr-3" />,
  },
];

function ExportBtn() {
  const doExport = (format: FormatType) => {
    const curitems = useSift.getState().items;
    const content = exportData(curitems, format);
    downloadContent(content, format);
  };
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline">
          <FileDown className="h-4 w-4 mr-1" />
          Export
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-32">
        <DropdownMenuLabel>Export format</DropdownMenuLabel>
        {ExportFormats.map((format, idx) => (
          <div key={idx}>
            <DropdownMenuItem
              key={format.name}
              onMouseDown={() => doExport(format.ext)}
            >
              {format.icon} {format.name}
            </DropdownMenuItem>
          </div>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function StatusFilterSelect() {
  const { setFiltering } = useSift();
  const [filtGroupSel, setFiltGroupSel] =
    useState<keyof typeof FiltPresets>("All");

  return (
    <div>
      <Select
        value={filtGroupSel}
        onValueChange={(val) => {
          setFiltGroupSel(val as keyof typeof FiltPresets);
          setFiltering([
            {
              id: "status",
              value: FiltPresets[val as keyof typeof FiltPresets],
            },
          ]);
        }}
      >
        <SelectTrigger className="inline-flex items-center gap-2">
          <Filter className="size-4" /> {sttDescriptionsj[filtGroupSel]}
        </SelectTrigger>
        <SelectContent>
          {Object.entries(sttDescriptionsj).map(([key, value]) => (
            <SelectItem key={key} value={key}>
              {value}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

const ROWS_PER_PAGE = 5;

export function MiniDisplayTable({
  data,
  columns = columnsBase,
}: {
  data: TableItem[];
  columns?: ColumnDef<TableItem, any>[];
}) {
  const [pagination, setPagination] = useState({
    pageIndex: 0, //initial page index
    pageSize: 5, //default page size
  });
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    onPaginationChange: setPagination,
    state: { pagination },
  });

  return (
    <div className="rounded-md border mt-4">
      <Table className="h-36">
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <TableHead key={header.id}>
                  {header.isPlaceholder
                    ? null
                    : flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={columns.length} className="h-24 text-center">
                No results.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      <div className="flex justify-between p-2">
        <Button
          onClick={() => table.previousPage()}
          disabled={!table.getCanPreviousPage()}
          size={"sm"}
        >
          {"<"}
        </Button>
        <span>
          Page {table.getState().pagination.pageIndex + 1} of{" "}
          {table.getPageCount()}
        </span>
        <Button
          onClick={() => table.nextPage()}
          disabled={!table.getCanNextPage()}
          size={"sm"}
        >
          {">"}
        </Button>
      </div>
    </div>
  );
}
