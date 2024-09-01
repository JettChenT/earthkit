"use client";

// TODO: add collapse, hover highlights, etc

import { Tool, useStore } from "@/lib/store";
import {
  Earth,
  Satellite,
  CarTaxiFront,
  SearchCode,
  Glasses,
  ArrowLeftToLineIcon,
  PanelRightIcon,
  KeyboardIcon,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Profile from "./profile";
import { UsageBar } from "./usagebar";
import { useEKGlobals } from "@/lib/globals";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";
import { TooltipProvider } from "./ui/tooltip";
import { SiDiscord } from "@icons-pack/react-simple-icons";
import Kbd, { KbdContainer, MetaKey } from "./keyboard";

export type SideBarItem = {
  tool: Tool;
  display: string;
  tooltip: string;
  icon: React.ReactNode;
};

const ignoreList = ["/", "/agent"];

export const sideBarData: SideBarItem[] = [
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
    tool: "geoclip",
    display: "Geoestimation",
    tooltip: "Global Location-Estimation Tool",
    icon: <Earth className="size-4" />,
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
];

export function EKLogo({ expanded }: { expanded: boolean }) {
  return (
    <span>
      <span className="text-blue-700">E</span>
      {expanded && <span>arth</span>}
      <span className="text-green-700">K</span>
      {expanded && <span>it</span>}
    </span>
  );
}

export default function Sidebar() {
  const pathname = usePathname();
  let { sidebarExpanded, setSidebarExpanded } = useEKGlobals();
  if (ignoreList.includes(pathname)) {
    return null;
  }
  return (
    <TooltipProvider>
      <aside
        className={cn(
          "flex-initial py-5 flex flex-col justify-between h-full border-r",
          sidebarExpanded ? "w-56" : "w-20 pr-2"
        )}
      >
        <nav className="grid gap-1 items-start px-2 text-sm font-medium lg:px-4">
          <div
            className={cn(
              "flex justify-center items-center mb-3 gap-1",
              sidebarExpanded ? "ml-2" : "ml-0.5"
            )}
          >
            <Link href="/" className="text-2xl font-bold font-mono">
              <EKLogo expanded={sidebarExpanded} />
            </Link>
            {sidebarExpanded && (
              <Button
                variant="ghost"
                onClick={() => setSidebarExpanded(!sidebarExpanded)}
                toolTip="Collapse Sidebar"
                side="right"
              >
                <ArrowLeftToLineIcon className="size-5" />
              </Button>
            )}
          </div>
          {sideBarData.map((item) => (
            <Button
              key={item.tool}
              asChild
              variant="ghost"
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground transition-all hover:text-primary justify-start ${
                pathname === `/${item.tool}`
                  ? "bg-muted/20 border border-gray-200 text-primary ring-1 ring-input"
                  : ""
              }`}
              toolTip={!sidebarExpanded ? item.display : undefined}
              side="right"
            >
              <Link href={`/${item.tool}`}>
                <div
                  className={
                    sidebarExpanded
                      ? "size-4"
                      : "size-6 w-6 h-6 flex items-center justify-center"
                  }
                >
                  {item.icon}
                </div>
                {sidebarExpanded && item.display}
              </Link>
            </Button>
          ))}
        </nav>
        <div
          className={cn(
            "flex flex-col gap-2",
            sidebarExpanded ? "px-3" : "ml-2"
          )}
        >
          {!sidebarExpanded && (
            <Button
              variant={"ghost"}
              onClick={() => setSidebarExpanded(true)}
              className="flex items-center"
              toolTip="Expand Sidebar"
              side="right"
            >
              <PanelRightIcon className="size-4" />
            </Button>
          )}
          <Button
            variant={sidebarExpanded ? "outline" : "ghost"}
            onClick={() => {
              document.dispatchEvent(new CustomEvent("OpenKbar"));
            }}
            className="flex items-center gap-2"
            toolTip={
              <KbdContainer>
                <MetaKey />
                <Kbd>K</Kbd>
              </KbdContainer>
            }
            side="right"
          >
            <KeyboardIcon className="size-4" />
            {sidebarExpanded && "Command Palette"}
          </Button>
          <UsageBar />
          <Profile />
        </div>
      </aside>
    </TooltipProvider>
  );
}
