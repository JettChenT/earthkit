"use client";
import React from "react";
import { MapViewState } from "@deck.gl/core";
import { useStore } from "@/lib/store";
import GeoCLIPPanel from "./operations/geoclip";

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
            return <GeoCLIPPanel />;
          default:
            return <div>TODO</div>;
        }
      })()}
    </div>
  );
}
