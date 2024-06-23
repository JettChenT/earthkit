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
    tool: "sift",
    display: "Sift",
    tooltip: "Sift",
    icon: <Glasses className="size-4" />,
  },
  {
    tool: "osm",
    display: "Overpass Query",
    tooltip:
      "Query the OpenStreetmap database with natural language and intelligent suggestions",
    icon: <SearchCode className="size-4" />,
  },
  {
    tool: "streetview",
    display: "Street View",
    tooltip: "Locate a ground-level image based on street view imagery",
    icon: <CarTaxiFront className="size-4" />,
  },
  {
    tool: "satellite",
    display: "Satellite",
    tooltip: "Locate a ground-level image based on satellite imagery",
    icon: <Satellite className="size-4" />,
  },
  {
    tool: "geoclip",
    display: "Geoestimation",
    tooltip: "Global Location-Estimation Tool",
    icon: <Earth className="size-4" />,
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  if (pathname === "/") {
    return null;
  }
  return (
    <div className="flex-initial w-56 py-5 flex flex-col justify-between h-full">
      <nav className="grid items-start px-2 text-sm font-medium lg:px-4">
        <Link href="/" className="text-2xl font-bold mb-3 ml-2 font-mono">
          <span className="text-blue-700">E</span>
          <span>arth</span>
          <span className="text-green-700">K</span>
          <span>it</span>
        </Link>
        {sideBarData.map((item) => (
          <Link
            key={item.tool}
            href={`/${item.tool}`}
            className={`flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground transition-all hover:text-primary ${
              pathname === `/${item.tool}`
                ? "bg-gray-100 border border-gray-200 text-primary"
                : ""
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
