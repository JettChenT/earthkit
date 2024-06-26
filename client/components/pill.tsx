import { LabelType } from "@/app/sift/siftStore";
import { cn, formatValue, isnil } from "@/lib/utils";
import { Skeleton } from "./ui/skeleton";
import { ReactNode } from "react";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { TooltipTrigger } from "@radix-ui/react-tooltip";

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
  onClick,
  tooltip,
  toolTipSide,
  className,
}: {
  children: React.ReactNode;
  color: PillColor;
  icon?: React.ReactNode;
  iconPosition?: "start" | "end";
  onClick?: () => void;
  tooltip?: ReactNode;
  toolTipSide?: "left" | "right" | "top" | "bottom";
  className?: string;
}) {
  let res = (
    <div
      className={cn(
        "px-2 py-1 rounded-md font-bold transition-all",
        getPillColorCn(color),
        className
      )}
      onClick={onClick}
    >
      {icon && iconPosition === "start" && <span className="mr-2">{icon}</span>}
      {children}
      {icon && iconPosition === "end" && <span className="ml-2">{icon}</span>}
    </div>
  );
  if (tooltip) {
    res = (
      <Tooltip>
        <TooltipTrigger>{res}</TooltipTrigger>
        <TooltipContent side={toolTipSide || "left"}>
          <div className="max-w-48">{tooltip}</div>
        </TooltipContent>
      </Tooltip>
    );
  }
  return res;
}

export function NumberPill({
  value,
  zval,
  baseColor,
  isFunCall,
  tooltip,
  toolTipSide,
}: {
  value: number;
  zval?: number;
  baseColor: PillColor;
  isFunCall?: boolean;
  tooltip?: ReactNode;
  toolTipSide?: "left" | "right" | "top" | "bottom";
}) {
  const color = zval && colorFromZVal(zval);
  return (
    <Pill color={baseColor} tooltip={tooltip} toolTipSide={toolTipSide}>
      <span style={color ? { color } : undefined}>
        <LoadingIfNull value={formatValue(value)} activated={isFunCall} />
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
  displayOverride,
}: {
  value: React.ReactNode;
  displayOverride?: React.ReactNode;
  activated?: boolean;
}) {
  return isnil(value) && activated ? (
    <Skeleton className="h-4 w-20" />
  ) : (
    displayOverride || value
  );
}
