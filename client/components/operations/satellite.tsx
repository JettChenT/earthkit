"use client";
import { Button } from "@/components/ui/button";
import { MAPBOX_TOKEN } from "@/lib/constants";
import { DeckGL, MapViewState } from "deck.gl";
import { useMemo, useRef, useState } from "react";
import { Map, MapRef } from "react-map-gl";
import { INITIAL_VIEW_STATE } from "../map";
import OperationContainer from "./ops";
import {
  EditableGeoJsonLayer,
  ViewMode,
  DrawRectangleMode,
  FeatureCollection,
} from "@deck.gl-community/editable-layers";

const selectedFeatureIndexes: number[] = [];

export default function Satellite() {
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState(false);
  const [featCollection, setFeatCollection] = useState<FeatureCollection>({
    type: "FeatureCollection",
    features: [],
  });
  const mapRef = useRef<MapRef>(null);
  const viewMode = useMemo(() => {
    return selecting ? DrawRectangleMode : ViewMode;
  }, [selecting]);

  const layer = new EditableGeoJsonLayer({
    id: "geojson-layer",
    data: featCollection,
    mode: viewMode,
    selectedFeatureIndexes,
    onEdit: ({ updatedData }) => {
      setFeatCollection(updatedData);
    },
  });

  return (
    <div>
      <div
        className={`absolute w-full h-full ${
          selecting ? "cursor-crosshair" : ""
        }`}
      >
        <DeckGL
          initialViewState={INITIAL_VIEW_STATE}
          controller
          layers={[layer]}
          getCursor={(st) => {
            if (selecting) return "crosshair";
            if (st.isDragging) return "grabbing";
            return "grab";
          }}
        >
          <Map
            mapboxAccessToken={MAPBOX_TOKEN}
            mapStyle="mapbox://styles/mapbox/satellite-v9"
            ref={mapRef}
          ></Map>
        </DeckGL>
      </div>
      <OperationContainer>
        {selecting ? (
          <div className="flex flex-row gap-2">
            <Button
              onClick={() => {
                setSelected(true);
                setSelecting(false);
                console.log(featCollection);
              }}
            >
              Done
            </Button>
            <Button onClick={() => setSelecting(false)} variant="secondary">
              Cancel
            </Button>
          </div>
        ) : (
          <Button onClick={() => setSelecting(true)}>Select Area</Button>
        )}
      </OperationContainer>
    </div>
  );
}
