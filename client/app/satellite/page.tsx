"use client";
import { Button } from "@/components/ui/button";
import { API_URL, MAPBOX_TOKEN } from "@/lib/constants";
import { Bounds, Coords, Point } from "@/lib/geo";
import {
  DrawRectangleMode,
  EditableGeoJsonLayer,
  FeatureCollection,
  ViewMode,
} from "@deck.gl-community/editable-layers";
import {
  DeckGL,
  DeckGLRef,
  MapViewState,
  PickingInfo,
  ScatterplotLayer,
} from "deck.gl";
import { useCallback, useMemo, useRef, useState } from "react";
import { Map, MapRef } from "react-map-gl";
import { INITIAL_VIEW_STATE } from "@/lib/constants";
import LatLngDisplay from "@/components/widgets/InfoBar";
import ImageUpload from "@/components/widgets/imageUpload";
import OperationContainer from "@/components/widgets/ops";
import dynamic from "next/dynamic";
const ESearchBox = dynamic(() => import("@/components/widgets/searchBox"), {
  ssr: false,
});
import "mapbox-gl/dist/mapbox-gl.css";

const selectedFeatureIndexes: number[] = [];

export default function Satellite() {
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState(false);
  const [image, setImage] = useState<string | null>(null);
  const [locating, setLocating] = useState(false);
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
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

  const getTooltip = useCallback(({ object }: PickingInfo<Point>) => {
    if (!object?.lat) return null;
    return object
      ? `Coordinates: ${object.lat.toFixed(4)}, ${object.lon.toFixed(4)}
      Confidence: ${object.aux.sim.toFixed(2)}
      Click to copy full coordinates`
      : null;
  }, []);

  const onLocate = () => {
    setLocating(true);
    setSelected(true);
    setSelecting(false);
    console.log(featCollection);
    const bounds: Bounds = {
      lo: {
        // @ts-ignore
        lat: featCollection.features[0].geometry.coordinates[0][0][1],
        // @ts-ignore
        lon: featCollection.features[0].geometry.coordinates[0][0][0],
        aux: {},
      },
      hi: {
        // @ts-ignore
        lat: featCollection.features[0].geometry.coordinates[0][2][1],
        // @ts-ignore
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

  const deckRef = useRef<DeckGLRef>(null);

  return (
    <div>
      <div
        className={`absolute w-full h-full ${
          selecting ? "cursor-crosshair" : ""
        }`}
      >
        <DeckGL
          ref={deckRef}
          initialViewState={viewState}
          controller
          layers={[layer, resultsLayer]}
          getTooltip={getTooltip}
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
          {selecting || selected ? (
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
                  setResults(null);
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
      <ESearchBox setViewState={setViewState} dglref={deckRef} />
    </div>
  );
}
