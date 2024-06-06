import { cn } from "@/lib/utils";

export type PillColor = "red" | "blue" | "green" | "orange" | "grey";

function getCn(color: PillColor) {
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
  }
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
    <div className={cn("px-2 py-1 rounded-md font-bold", getCn(color))}>
      {icon && iconPosition === "start" && <span className="mr-2">{icon}</span>}
      {children}
      {icon && iconPosition === "end" && <span className="ml-2">{icon}</span>}
    </div>
  );
}
