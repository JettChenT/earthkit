import { TableItem } from "@/app/sift/siftStore";
import { ColumnDef, StringOrTemplateHeader } from "@tanstack/react-table";
import {
  ArrowDown01,
  ArrowDown10,
  ArrowDownAZ,
  ArrowDownZA,
  ArrowUp01,
  ArrowUpAZ,
  ArrowUpDown,
} from "lucide-react";
import { columnHelper } from "./table";
import Pill, {
  LoadingIfNull,
  NumberPill,
  PillColor,
  StatusCell,
  StatusNumberMap,
} from "@/components/pill";
import { formatValue, isnil } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { accessProperty } from "./inout";

// hard-coded columns
export type CoordCol = {
  type: "CoordCol";
  accessor: "coord";
  header: "Coords";
};

export type StatusCol = {
  type: "StatusCol";
  accessor: "status";
  header: "Status";
};

// For custom-defined columns
export type ColBase = {
  accessor: string;
  tooltipAccessor?: string;
  header: string;
  isFunCall?: boolean;
};

export type NumericalCol = ColBase & {
  type: "NumericalCol";
  mean?: number;
  stdev?: number;
  baseColor?: PillColor;
};

export type TextCol = ColBase & {
  type: "TextCol";
  accessor: string;
  usePill?: boolean;
  baseColor?: PillColor;
};

export type BoolCol = ColBase & {
  type: "BoolCol";
  accessor: string;
  isFunCall?: boolean;
};

const sortableHeader: (
  headerName: string,
  typ: "text" | "num"
) => StringOrTemplateHeader<TableItem, any> = (headerName, typ) => {
  return function SortableHeader({ column }) {
    const icn = (() => {
      const cname = "ml-2 h-4 w-4";
      switch (column.getIsSorted()) {
        case false:
          return <ArrowUpDown className={cname + " text-gray-400"} />;
        case "asc":
          return typ == "text" ? (
            <ArrowUpAZ className={cname} />
          ) : (
            <ArrowUp01 className={cname} />
          );
        case "desc":
          return typ == "text" ? (
            <ArrowDownZA className={cname} />
          ) : (
            <ArrowDown10 className={cname} />
          );
      }
    })();
    return (
      <Button variant="ghost" onClick={() => column.toggleSorting()}>
        {headerName} {icn}
      </Button>
    );
  };
};

export function compileColDefs(cols: Col[]): ColumnDef<TableItem, any>[] {
  return cols.map((col) => {
    switch (col.type) {
      case "CoordCol":
        return columnHelper.accessor("coord", {
          header: "Coords",
          cell: (props) => {
            const coord = props.getValue();
            return (
              <Pill
                color="blue"
                onClick={() => {
                  navigator.clipboard.writeText(`${coord.lat}, ${coord.lon}`);
                  toast.success("Copied Coordinates to clipboard");
                }}
              >
                {formatValue(coord.lat)}, {formatValue(coord.lon)}
              </Pill>
            );
          },
        });
      case "StatusCol":
        return columnHelper.accessor("status", {
          header: sortableHeader("Status", "text"),
          cell: (props) => {
            const status = props.getValue();
            return <StatusCell status={status} />;
          },
          sortingFn: (a, b, cid) => {
            return (
              StatusNumberMap[a.original.status] -
              StatusNumberMap[b.original.status]
            );
          },
          filterFn: (row, columnFilter, filterValue) => {
            return filterValue.includes(row.getValue(columnFilter));
          },
        });
      case "NumericalCol":
        return columnHelper.accessor(`aux.${col.accessor}`, {
          header: sortableHeader(col.header, "num"),
          cell: (props) => {
            const val = props.getValue();
            console.log(props);
            let tooltip = col.tooltipAccessor
              ? accessProperty(props.row.original.aux, col.tooltipAccessor)
              : undefined;
            return (
              <NumberPill
                value={val}
                zval={
                  col.mean && col.stdev
                    ? (val - col.mean) / col.stdev
                    : undefined
                }
                baseColor={col.baseColor || "hidden"}
                tooltip={tooltip}
                isFunCall={col.isFunCall}
              />
            );
          },
        });
      case "TextCol":
        return columnHelper.accessor(`aux.${col.accessor}`, {
          header: sortableHeader(col.header, "text"),
          cell: (props) => {
            const val = props.getValue();
            let tooltip = col.tooltipAccessor
              ? accessProperty(props.row.original.aux, col.tooltipAccessor)
              : undefined;
            const valNode = (
              <LoadingIfNull value={val} activated={col.isFunCall} />
            );
            return (
              <Pill color={col.baseColor || "hidden"} tooltip={tooltip}>
                {valNode}
              </Pill>
            );
          },
        });
      case "BoolCol":
        return columnHelper.accessor(`aux.${col.accessor}`, {
          header: sortableHeader(col.header, "text"),
          cell: (props) => {
            const val = props.getValue();
            let tooltip = col.tooltipAccessor
              ? accessProperty(props.row.original.aux, col.tooltipAccessor)
              : undefined;
            return (
              <Pill
                color={isnil(val) ? "hidden" : val ? "green" : "red"}
                tooltip={tooltip}
              >
                <LoadingIfNull
                  displayOverride={val ? "yes" : "no"}
                  value={val}
                  activated={col.isFunCall}
                />
              </Pill>
            );
          },
        });
    }
  });
}

export const defaultCols: Col[] = [
  { type: "CoordCol", accessor: "coord", header: "Coords" },
  { type: "StatusCol", accessor: "status", header: "Status" },
];
export const defaultColDefs: ColumnDef<TableItem, any>[] =
  compileColDefs(defaultCols);

export function mergeCols(colA: Col[], colB: Col[]) {
  const cbfilt = colB.filter(
    (col) =>
      colA.find(
        (acol) => acol.accessor === col.accessor && acol.type === col.type
      ) === undefined
  );
  return [...colA, ...cbfilt];
}

export type Col = CoordCol | StatusCol | NumericalCol | TextCol | BoolCol;
