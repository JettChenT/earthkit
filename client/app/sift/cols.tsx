import { TableItem } from "@/app/sift/siftStore";
import { ColumnDef } from "@tanstack/react-table";
import { columnHelper } from "./table";
import Pill, {
  NumberPill,
  PillColor,
  StatusCell,
  StatusNumberMap,
} from "@/components/pill";
import { formatValue } from "@/lib/utils";

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
          header: "Status",
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
          header: col.header,
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
          header: col.header,
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
