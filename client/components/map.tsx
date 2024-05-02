"use client";
import React from "react";
import DeckGL from "@deck.gl/react";
import { MapView, MapViewState } from "@deck.gl/core";
import { LineLayer } from "@deck.gl/layers";
import { Map } from "react-map-gl";

const INITIAL_VIEW_STATE: MapViewState = {
  longitude: -122.41669,
  latitude: 37.7853,
  zoom: 13,
};

export default function MapDisplay() {
  console.log(process.env.NEXT_PUBLIC_MAPBOX_TOKEN);
  return (
    <div className="h-full flex-1 relative">
      <DeckGL initialViewState={INITIAL_VIEW_STATE} controller>
        <Map
          mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
          mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
        />
      </DeckGL>
    </div>
  );
}
