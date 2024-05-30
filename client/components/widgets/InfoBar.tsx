import React from "react";
import { Point } from "@/lib/geo";

interface LatLngDisplayProps {
  cursorCoords: Point;
}

const LatLngDisplay: React.FC<LatLngDisplayProps> = ({ cursorCoords }) => {
  return (
    <div className="absolute bottom-8 right-3 bg-white p-3 rounded-md bg-opacity-80 font-mono">
      <div>Lat: {cursorCoords.lat.toFixed(8)}</div>
      <div>Lon: {cursorCoords.lon.toFixed(8)}</div>
    </div>
  );
};

export default LatLngDisplay;
