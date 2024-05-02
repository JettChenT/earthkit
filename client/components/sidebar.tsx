"use client";

import { Tool, useStore } from "@/lib/store";
import { Earth, Satellite, CarTaxiFront } from "lucide-react";

export type SideBarItem = {
  tool: Tool;
  display: string;
  tooltip: string;
  icon: React.ReactNode;
};

const sideBarData: SideBarItem[] = [
  {
    tool: "geoclip",
    display: "GeoCLIP",
    tooltip: "Global Location-Estimation Tool",
    icon: <Earth className="size-4" />,
  },
  {
    tool: "satellite",
    display: "Satellite",
    tooltip: "Locate a ground-level image based on satellite imagery",
    icon: <Satellite className="size-4" />,
  },
  {
    tool: "streetview",
    display: "Street View",
    tooltip: "Locate a ground-level image based on street view imagery",
    icon: <CarTaxiFront className="size-4" />,
  },
];

export const Sidebar = () => {
  const { tool, setTool } = useStore();
  return (
    <div className="flex-1">
      <nav className="grid items-start px-2 text-sm font-medium lg:px-4">
        {sideBarData.map((item) => (
          <button
            key={item.tool}
            className={`flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground transition-all hover:text-primary ${
              tool === item.tool ? "bg-gray-100" : ""
            }`}
            onClick={() => setTool(item.tool)}
          >
            {item.icon}
            {item.display}
          </button>
        ))}
      </nav>
    </div>
  );
};
