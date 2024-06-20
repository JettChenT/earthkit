import { LabelType } from "@/app/sift/siftStore";
import { cn, formatValue } from "@/lib/utils";
import { Skeleton } from "./ui/skeleton";

export type PillColor = "red" | "blue" | "green" | "orange" | "grey" | "hidden";

export function getPillColorCn(color: PillColor) {
  switch (color) {
    case "red":
      return "bg-red-100 text-red-500";
    case "blue":
      return "bg-blue-100 text-blue-500";
    case "green":
      return "bg-green-100 text-green-500";
    case "orange":
      return "bg-orange-100 text-orange-500";
    case "grey":
      return "bg-slate-200 text-slate-500";
    case "hidden":
      return "bg-transparent text-current";
  }
}

function colorFromZVal(z: number) {
  const clampedZ = Math.max(-1, Math.min(2, z));
  const normalizedZ = (clampedZ + 1) / 3;
  const r = Math.round(255 * normalizedZ);
  const g = 0;
  const b = Math.round(255 * (1 - normalizedZ));
  return `rgb(${r},${g},${b})`;
}

export default function Pill({
  children,
  color,
  icon,
  iconPosition = "start", // default position is "start"
}: {
  children: React.ReactNode;
  color: PillColor;
  icon?: React.ReactNode;
  iconPosition?: "start" | "end";
}) {
  return (
    <div
      className={cn("px-2 py-1 rounded-md font-bold", getPillColorCn(color))}
    >
      {icon && iconPosition === "start" && <span className="mr-2">{icon}</span>}
      {children}
      {icon && iconPosition === "end" && <span className="ml-2">{icon}</span>}
    </div>
  );
}

export function NumberPill({
  value,
  zval,
  baseColor,
  isFunCall,
}: {
  value: number;
  zval: number;
  baseColor: PillColor;
  isFunCall?: boolean;
}) {
  const color = colorFromZVal(zval);
  return (
    <Pill color={baseColor}>
      <span style={{ color }}>
        <LoadingIfNull value={formatValue(value)} activated={!isFunCall} />
      </span>
    </Pill>
  );
}

export const statusToPillColor = (status: LabelType): PillColor => {
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

export const StatusNumberMap: Record<LabelType, number> = {
  "Not Labeled": 0,
  Match: 1,
  Keep: 2,
  "Not Match": 3,
};

export function StatusCell({ status }: { status: LabelType }) {
  const col: PillColor = statusToPillColor(status);
  return <Pill color={col}>{status}</Pill>;
}

export function LoadingIfNull({
  value,
  activated,
}: {
  value: React.ReactNode;
  activated?: boolean;
}) {
  return value || !activated ? value : <Skeleton className="h-4 w-20" />;
}
