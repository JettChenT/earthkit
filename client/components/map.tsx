"use client";
import React from "react";
import { MapViewState } from "@deck.gl/core";
import { useStore } from "@/lib/store";
import GeoCLIP from "./operations/geoclip";
import Satellite from "./operations/satellite";
import StreetView from "./operations/streetview";

export const INITIAL_VIEW_STATE: MapViewState = {
  longitude: -122.41669,
  latitude: 37.7853,
  zoom: 1,
};

export default function MapDisplay() {
  let { tool } = useStore();
  return (
    <div className="h-full flex-1 relative">
      {(() => {
        switch (tool) {
          case "geoclip":
            return <GeoCLIP />;
          case "satellite":
            return <Satellite />;
          case "streetview":
            return <StreetView />;
          default:
            return null;
        }
      })()}
    </div>
  );
}
