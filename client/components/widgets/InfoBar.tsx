import React from "react";
import { Point } from "@/lib/geo";
import Kbd, { MetaKey } from "../keyboard";
import { cn } from "@/lib/utils";

interface InfoBarProps {
  cursorCoords: Point;
  showShortcuts?: boolean;
  className?: string;
}

const InfoBar: React.FC<InfoBarProps> = ({
  cursorCoords,
  showShortcuts,
  className,
}) => {
  return (
    <div
      className={cn(
        "absolute bottom-12 right-4 bg-white p-3 rounded-md bg-opacity-70 font-mono border",
        className
      )}
    >
      <div>Lat: {cursorCoords.lat.toFixed(8)}</div>
      <div>Lon: {cursorCoords.lon.toFixed(8)}</div>
    </div>
  );
};

export default InfoBar;
