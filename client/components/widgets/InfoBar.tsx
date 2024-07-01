import React from "react";
import { Point } from "@/lib/geo";
import Kbd, { MetaKey } from "../keyboard";
import { cn } from "@/lib/utils";

interface LatLngDisplayProps {
  cursorCoords: Point;
  showShortcuts?: boolean;
  className?: string;
}

const LatLngDisplay: React.FC<LatLngDisplayProps> = ({
  cursorCoords,
  showShortcuts,
  className,
}) => {
  return (
    <div
      className={cn(
        "absolute bottom-8 right-3 bg-white p-3 rounded-md bg-opacity-70 font-mono border",
        className
      )}
    >
      <div>Lat: {cursorCoords.lat.toFixed(8)}</div>
      <div>Lon: {cursorCoords.lon.toFixed(8)}</div>
      {showShortcuts && (
        <div className="flex items-center gap-2 text-sm">
          Copy Coords <MetaKey /> <Kbd>C</Kbd>
        </div>
      )}
    </div>
  );
};

export default LatLngDisplay;
