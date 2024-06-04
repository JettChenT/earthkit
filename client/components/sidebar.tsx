"use client";

// TODO: add collapse, hover highlights, etc

import { Tool, useStore } from "@/lib/store";
import {
  Earth,
  Satellite,
  CarTaxiFront,
  SearchCode,
  Glasses,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Profile from "./profile";

export type SideBarItem = {
  tool: Tool;
  display: string;
  tooltip: string;
  icon: React.ReactNode;
};

const sideBarData: SideBarItem[] = [
  {
    tool: "osm",
    display: "OSM",
    tooltip: "OpenStreetMap",
    icon: <SearchCode className="size-4" />,
  },
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
  {
    tool: "comb",
    display: "Comb",
    tooltip: "Comb",
    icon: <Glasses className="size-4" />,
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  return (
    <div className="flex-initial w-56 py-5 flex flex-col justify-between h-full">
      <nav className="grid items-start px-2 text-sm font-medium lg:px-4">
        <h1 className="text-2xl font-bold mb-3 ml-2">EarthKit</h1>
        {sideBarData.map((item) => (
          <Link
            key={item.tool}
            href={`/${item.tool}`}
            className={`flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground transition-all hover:text-primary ${
              pathname === `/${item.tool}` ? "bg-gray-100" : ""
            }`}
          >
            {item.icon}
            {item.display}
          </Link>
        ))}
      </nav>
      <Profile />
    </div>
  );
}
