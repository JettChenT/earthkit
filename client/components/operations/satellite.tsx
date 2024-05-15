"use client";
import { Button } from "@/components/ui/button";
import { API_URL, MAPBOX_TOKEN } from "@/lib/constants";
import { DeckGL, MapViewState, ScatterplotLayer } from "deck.gl";
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
import { Point, Bounds, Coords } from "@/lib/geo";
import LatLngDisplay from "./widgets/InfoBar";
import ImageUpload from "./widgets/imageUpload";

const selectedFeatureIndexes: number[] = [];

export default function Satellite() {
  const [selecting, setSelecting] = useState(false);
  const [image, setImage] = useState<string | null>(null);
  const [locating, setLocating] = useState(false);
  const [featCollection, setFeatCollection] = useState<FeatureCollection>({
    type: "FeatureCollection",
    features: [],
  });
  const [cursorCoords, setCursorCoords] = useState<Point>({
    lat: 0,
    lon: 0,
    aux: null,
  });
  const [results, setResults] = useState<Coords | null>(null);
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
      setFeatCollection({
        type: "FeatureCollection",
        features: updatedData.features.slice(-1),
      });
    },
  });

  const resultsLayer = new ScatterplotLayer({
    id: "results-layer",
    data: results?.coords,
    getPosition: (d) => [d.lon, d.lat],
    getRadius: (d) => 3,
    getFillColor: (d) => [255 * d.aux.sim, 140, 0],
    pickable: true,
    radiusScale: 1,
    radiusMinPixels: 5,
    radiusMaxPixels: 100,
  });

  const onLocate = () => {
    setLocating(true);
    setSelecting(false);
    console.log(featCollection);
    const bounds: Bounds = {
      lo: {
        lat: featCollection.features[0].geometry.coordinates[0][0][1],
        lon: featCollection.features[0].geometry.coordinates[0][0][0],
        aux: {},
      },
      hi: {
        lat: featCollection.features[0].geometry.coordinates[0][2][1],
        lon: featCollection.features[0].geometry.coordinates[0][2][0],
        aux: {},
      },
    };
    fetch(`${API_URL}/satellite/locate`, {
      method: "POST",
      body: JSON.stringify({
        bounds,
        image_url: image,
      }),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        setResults(data);
        setLocating(false);
      });
  };

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
          layers={[layer, resultsLayer]}
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
                disabled={featCollection.features.length === 0 || locating}
                onClick={() => {
                  console.log(featCollection);
                  onLocate();
                }}
              >
                {locating ? "Locating..." : "Locate"}
              </Button>
              <Button
                onClick={() => {
                  setSelecting(false);
                  setFeatCollection({
                    type: "FeatureCollection",
                    features: [],
                  });
                }}
                variant="secondary"
              >
                Cancel Selection
              </Button>
            </div>
          ) : (
            <Button onClick={() => setSelecting(true)}>
              Select Search Range
            </Button>
          )}
        </div>
      </OperationContainer>
      <LatLngDisplay cursorCoords={cursorCoords} />
    </div>
  );
}
