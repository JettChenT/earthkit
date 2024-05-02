"use client";
import React, { useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { MapView, MapViewState } from "@deck.gl/core";
import { ScatterplotLayer } from "@deck.gl/layers";
import { Map } from "react-map-gl";
import Operations from "./ops";
import { useStore } from "@/lib/store";
import { Coords, Point } from "@/lib/geo";

const INITIAL_VIEW_STATE: MapViewState = {
  longitude: -122.41669,
  latitude: 37.7853,
  zoom: 1,
};

export default function MapDisplay() {
  let { layers } = useStore();

  return (
    <div className="h-full flex-1 relative">
      <DeckGL initialViewState={INITIAL_VIEW_STATE} controller>
        <Map
          mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
          mapboxAccessToken={process.env.NEXT_PUBLIC_MAPBOX_TOKEN}
        />
        {layers.map((layer) => {
          return (
            <ScatterplotLayer
              id={layer.id}
              data={layer.coords}
              getPosition={(d) => [d.lat, d.lon]}
              getRadius={() => 0.1}
              getFillColor={(d) => {
                return [d.aux.conf * 255, 0, 0];
              }}
              pickable={true}
              radiusScale={1}
              radiusMinPixels={5}
              radiusMaxPixels={100}
            />
          );
        })}
        <Operations />
      </DeckGL>
    </div>
  );
}
