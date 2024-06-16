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
  NumberPill,
  PillColor,
  StatusCell,
  StatusNumberMap,
} from "@/components/pill";
import { formatValue } from "@/lib/utils";
import { Button } from "@/components/ui/button";

// hard-coded columns
export type CoordCol = {
  type: "CoordCol";
  accessor: "coord";
};

export type StatusCol = {
  type: "StatusCol";
  accessor: "status";
};

// For custom-defined columns
export type ColBase = {
  accessor: string;
  header: string;
};

export type NumericalCol = ColBase & {
  type: "NumericalCol";
  mean: number;
  stdev: number;
  baseColor?: PillColor;
};

export type TextCol = ColBase & {
  type: "TextCol";
  accessor: string;
  usePill?: boolean;
  baseColor?: PillColor;
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
              <Pill color="blue">
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
            return (
              <NumberPill
                value={val}
                zval={(val - col.mean) / col.stdev}
                baseColor={col.baseColor || "grey"}
              />
            );
          },
        });
      case "TextCol":
        return columnHelper.accessor(`aux.${col.accessor}`, {
          header: sortableHeader(col.header, "text"),
          cell: (props) => {
            const val = props.getValue();
            if (col.usePill) {
              return <Pill color={col.baseColor || "grey"}>{val}</Pill>;
            } else {
              return <span>{val}</span>;
            }
          },
        });
    }
  });
}

export const defaultCols: Col[] = [
  { type: "CoordCol", accessor: "coord" },
  { type: "StatusCol", accessor: "status" },
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

export type Col = CoordCol | StatusCol | NumericalCol | TextCol;
