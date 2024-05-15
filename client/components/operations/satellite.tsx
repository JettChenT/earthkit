"use client";
import { Button } from "@/components/ui/button";
import { MAPBOX_TOKEN } from "@/lib/constants";
import { DeckGL, MapViewState } from "deck.gl";
import { useMemo, useRef, useState } from "react";
import { Map, MapRef } from "react-map-gl";
import { INITIAL_VIEW_STATE } from "../map";
import OperationContainer from "./widgets/ops";
import {
  EditableGeoJsonLayer,
  ViewMode,
  DrawRectangleMode,
  FeatureCollection,
} from "@deck.gl-community/editable-layers";
import { Point } from "@/lib/geo";
import LatLngDisplay from "./widgets/InfoBar";
import ImageUpload from "./widgets/imageUpload";

const selectedFeatureIndexes: number[] = [];

export default function Satellite() {
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState(false);
  const [image, setImage] = useState<string | null>(null);
  const [featCollection, setFeatCollection] = useState<FeatureCollection>({
    type: "FeatureCollection",
    features: [],
  });
  const [cursorCoords, setCursorCoords] = useState<Point>({
    lat: 0,
    lon: 0,
    aux: null,
  });
  const mapRef = useRef<MapRef>(null);
  const viewMode = useMemo(() => {
    let vm = selecting ? DrawRectangleMode : ViewMode;
    vm.prototype.handlePointerMove = ({ mapCoords }) => {
      setCursorCoords({
        lon: mapCoords[0],
        lat: mapCoords[1],
        aux: null,
      });
    };
    return vm;
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
            mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
            ref={mapRef}
          ></Map>
        </DeckGL>
      </div>
      <OperationContainer className="bg-opacity-85">
        <article className="prose prose-sm leading-5 mb-3">
          <h3>Satellite Geolocalization</h3>
          Select an area on the map to get the crossview location of the
          satellite. This uses the{" "}
          <a href="https://paperswithcode.com/paper/sample4geo-hard-negative-sampling-for-cross">
            Sample4Geo
          </a>{" "}
          model.
        </article>
        <div className="flex flex-col gap-2">
          <ImageUpload
            onSetImage={setImage}
            image={image}
            className="border-stone-400"
          />
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
        </div>
      </OperationContainer>
      <LatLngDisplay cursorCoords={cursorCoords} />
    </div>
  );
}
