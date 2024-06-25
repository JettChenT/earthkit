"use client";
import {
  ColumnDef,
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { FiltPresets, TableItem, useSift } from "./siftStore";

import { Button } from "@/components/ui/button";
import { Dialog, DialogTrigger } from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { API_URL } from "@/lib/constants";
import { PointFromPurePoint } from "@/lib/geo";
import { ingestStream } from "@/lib/rpc";
import { downloadContent } from "@/lib/utils";
import ky from "ky";
import {
  CarFront,
  DotIcon,
  FileDown,
  FileInput,
  FileJson2,
  Filter,
  Globe,
  MapPin,
  Plus,
  Satellite,
  SearchCode,
  Table as TableIcn,
  Trash,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { toast } from "sonner";
import { Key } from "ts-key-enum";
import { compileColDefs, defaultColDefs } from "./cols";
import { GeoImport } from "./geo-import";
import { FormatType, exportData } from "./inout";
import { CustomExtraction } from "./lmm";
import { useRouter } from "next/navigation";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CommandBar, Listener, useListeners } from "@/components/kbar";
import { useKy } from "@/lib/api";
import { useSWRConfig } from "swr";

export const columnHelper = createColumnHelper<TableItem>();

const sttDescriptionsj = {
  All: "All",
  Labeled: "Labeled",
  NotLabeled: "Not Labeled",
  Match: "Match",
  Keep: "Keep",
  NotMatch: "Not Match",
};

export default function SiftTable() {
  let {
    items,
    idx,
    setIdx,
    sorting,
    setSorting,
    filtering,
    setFiltering,
    cols,
  } = useSift();
  let [selectedIdx, setSelectedIdx] = useState(0);
  const tableContainerRef = useRef<HTMLDivElement | null>(null);
  let colDefs = useMemo(() => {
    return compileColDefs(cols);
  }, [cols]);

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
      <TooltipProvider>
        <div className="flex justify-end gap-2 items-center">
          <ActionBtn />
          <Tooltip>
            <TooltipTrigger asChild>
              <ImportBtn />
            </TooltipTrigger>
            <TooltipContent>
              <p>Import Items</p>
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <ExportBtn />
            </TooltipTrigger>
            <TooltipContent>
              <p>Export Data</p>
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <ClearBtn />
            </TooltipTrigger>
            <TooltipContent>
              <p>Clear Table</p>
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <StatusFilterSelect />
            </TooltipTrigger>
            <TooltipContent>
              <p>Filter table data by status</p>
            </TooltipContent>
          </Tooltip>
        </div>
        <div
          ref={tableContainerRef}
          className="h-full overflow-auto rounded-sm border"
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
              {items.length == 0 ? (
                <TableRow className="hover:bg-muted/0">
                  <TableCell colSpan={cols.length} className="p-4">
                    <GetStarted />
                  </TableCell>
                </TableRow>
              ) : table.getRowModel().rows?.length ? (
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
                  <TableCell colSpan={cols.length} className="h-24 text-center">
                    No results.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
        <CommandBar
          commands={[
            {
              type: "CommandGroupData",
              heading: "Sift Table Actions",
              children: [
                {
                  type: "CommandItemData",
                  event: "SiftImport",
                  display: "Import File",
                  icon: <FileInput className="size-5" />,
                },
                {
                  type: "CommandItemData",
                  event: "SiftExportCSV",
                  display: "Export CSV",
                  icon: <FileDown className="size-5" />,
                  disabled: !items.length,
                },
                {
                  type: "CommandItemData",
                  event: "SiftExportGeoJSON",
                  display: "Export GeoJSON",
                  icon: <Globe className="size-5" />,
                  disabled: !items.length,
                },
                {
                  type: "CommandItemData",
                  event: "SiftExportJSON",
                  display: "Export JSON",
                  icon: <FileJson2 className="size-5" />,
                  disabled: !items.length,
                },
                {
                  type: "CommandItemData",
                  event: "SiftClear",
                  display: "Clear Table",
                  icon: <Trash className="size-5" />,
                  disabled: !items.length,
                },
                {
                  type: "CommandItemData",
                  event: "SiftVLM",
                  display: "Add Feature: Vision-Language Model",
                  icon: <SearchCode className="size-5" />,
                  disabled: !items.length,
                },
                {
                  type: "CommandItemData",
                  event: "SiftGeoClip",
                  display: "Add Feature: GeoCLIP",
                  icon: <MapPin className="size-5" />,
                  disabled: !items.length,
                },
                {
                  type: "CommandItemData",
                  event: "SiftStreetview",
                  display: "Add Feature: Streetview",
                  icon: <CarFront className="size-5" />,
                  disabled: !items.length,
                },
                {
                  type: "CommandItemData",
                  event: "SiftSatellite",
                  display: "Add Feature: Satellite",
                  icon: <Satellite className="size-5" />,
                  disabled: !items.length,
                },
              ],
            },
          ]}
        />
      </TooltipProvider>
    </div>
  );
}

function GetStarted() {
  const router = useRouter();
  let [importOpen, setImportOpen] = useState(false);
  return (
    <div className="w-full -mt-14 prose lg:prose-lg">
      <p>No Coordinates to see yet!</p>
      <div className="flex flex-col gap-4 mx-auto not-prose w-full">
        <p className="-mb-2">Get Started:</p>
        <Dialog open={importOpen} onOpenChange={setImportOpen}>
          <DialogTrigger asChild>
            <GSCard
              title="Import File"
              description="Import geojson, csv, json..."
              icon={<FileInput className="size-5" />}
            />
          </DialogTrigger>
          <GeoImport setOpen={setImportOpen} />
        </Dialog>
        <GSCard
          title="Overpass Turbo Query"
          description="Natural-language querying with Intelligent Suggestions"
          icon={<SearchCode className="size-5" />}
          onClick={() => router.push("/osm")}
        />
        <GSCard
          title="Sample Streetview Locations (Experimental)"
          description="Start from sampling streetview imagery of an area"
          icon={<CarFront className="size-5" />}
          onClick={() => router.push("/streetview")}
        />
        <GSCard
          title="Sample Satellite Locations (Experimental)"
          description="Start from sampling Satellite imagery of an area"
          icon={<Satellite className="size-5" />}
          onClick={() => router.push("/satellite")}
        />
      </div>
    </div>
  );
}

function GSCard({
  title,
  description,
  icon,
  onClick,
}: {
  title: string;
  description: string;
  icon: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <Button
      variant={"secondary"}
      size={"lg"}
      className="max-w-md xl:max-w-lg justify-start rounded-lg border border-gray-200 shadow-sm flex h-14 flex-row gap-3 pl-3 hover:scale-[102%] transition-all"
      onClick={onClick}
    >
      <div className="flex-none">{icon}</div>
      <div className="flex-grow flex flex-col">
        <span className="text-[16px] font-bold w-full text-left">{title}</span>
        <span className="text-[14px] w-full text-left break-words">
          {description}
        </span>
      </div>
    </Button>
  );
}

function ImportBtn() {
  const [open, setOpen] = useState(false);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          onClick={() => setOpen(true)}
          regEvent="SiftImport"
        >
          <FileInput className="h-4 w-4" />
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

function ActionBtn() {
  const { target_image, cols, setCols, items, updateItemResults } = useSift();
  const getKyInst = useKy();
  const { mutate } = useSWRConfig();
  const similarityAction = async (
    actionName: "geoclip" | "streetview" | "satellite"
  ) => {
    console.log(actionName);
    if (!target_image) {
      toast.error("No target image uploaded");
      return;
    }
    const payload = {
      image_url: target_image,
      coords: {
        coords: items.map((item) => PointFromPurePoint(item.coord, {})),
      },
    };
    const targ_url = (() => {
      switch (actionName) {
        case "geoclip":
          return "geoclip/similarity/streaming";
        case "streetview":
          return "streetview/locate/streaming";
        case "satellite":
          return "satellite/similarity/streaming";
      }
    })();
    setCols((cols) => [
      ...cols,
      {
        type: "NumericalCol",
        accessor:
          actionName == "streetview" ? "streetview.max_sim" : actionName,
        header: actionName,
        mean: 0,
        stdev: 0.1,
      },
    ]);

    const kyInst = await getKyInst();
    const res = await kyInst.post(targ_url, {
      timeout: false,
      json: payload,
    });

    if (!res.ok || !res.body) {
      console.error(res);
      toast.error("Failed to get results");
      return;
    }

    mutate("/api/usage");

    for await (const chunk of ingestStream(res)) {
      if (chunk.type == "ResultsUpdate") {
        updateItemResults(chunk, actionName);
      }
    }
  };

  const extractionTrigger = useRef<HTMLDivElement>(null);

  useListeners([
    {
      event: "SiftVLM",
      handler: () => extractionTrigger.current?.click(),
    },
    {
      event: "SiftGeoClip",
      handler: () => similarityAction("geoclip"),
    },
    {
      event: "SiftStreetview",
      handler: () => similarityAction("streetview"),
    },
    {
      event: "SiftSatellite",
      handler: () => similarityAction("satellite"),
    },
  ]);

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="default" disabled={!items.length}>
            <Plus className="size-4 mr-1" />
            Feature
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuLabel>Similarity Scores</DropdownMenuLabel>
          <DropdownMenuItem onClick={() => similarityAction("geoclip")}>
            GeoCLIP
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => similarityAction("streetview")}>
            Streetview
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => similarityAction("satellite")}>
            Satellite
          </DropdownMenuItem>
          <DropdownMenuLabel>Custom Extraction</DropdownMenuLabel>
          <DropdownMenuItem
            onClick={() => {
              extractionTrigger.current?.click();
            }}
          >
            Vision-Language Model
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <CustomExtraction ref={extractionTrigger} />
    </>
  );
}

function ExportBtn() {
  const doExport = (format: FormatType) => {
    const curitems = useSift.getState().items;
    const content = exportData(
      { items: curitems, cols: useSift.getState().cols },
      format
    );
    downloadContent(content, format);
  };
  const listeners: Listener[] = [
    { event: "SiftExportCSV", handler: () => doExport("csv") },
    { event: "SiftExportGeoJSON", handler: () => doExport("geojson") },
    { event: "SiftExportJSON", handler: () => doExport("json") },
  ];
  useListeners(listeners);
  const { items } = useSift();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" disabled={!items.length}>
          <FileDown className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-32">
        <DropdownMenuLabel>Export format</DropdownMenuLabel>
        {ExportFormats.map((format, idx) => (
          <div key={idx}>
            <DropdownMenuItem
              key={format.name}
              onClick={() => doExport(format.ext)}
            >
              {format.icon} {format.name}
            </DropdownMenuItem>
          </div>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function ClearBtn() {
  const { clearTable, items } = useSift();
  return (
    <Button
      onClick={clearTable}
      variant={"outline"}
      disabled={!items.length}
      regEvent="SiftClear"
    >
      <Trash className="size-4" />
    </Button>
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

const ROWS_PER_PAGE = 4;

export function MiniDisplayTable({
  data,
  columns = defaultColDefs,
}: {
  data: TableItem[];
  columns?: ColumnDef<TableItem, any>[];
}) {
  const [pagination, setPagination] = useState({
    pageIndex: 0, //initial page index
    pageSize: ROWS_PER_PAGE, //default page size
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
    <div className="rounded-md border mt-4 overflow-x-scroll">
      <div className="h-[300px]">
        <Table>
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
                <TableRow key={row.id} className="h-7">
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
                  colSpan={columns.length}
                  className="h-24 text-center"
                >
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
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
