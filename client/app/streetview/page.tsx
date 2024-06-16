"use client";
import { Button } from "@/components/ui/button";
import { API_URL, MAPBOX_TOKEN, RAW_API_URL } from "@/lib/constants";
import { Bounds, Coords, Point } from "@/lib/geo";
import {
  DrawRectangleMode,
  EditableGeoJsonLayer,
  FeatureCollection,
  ViewMode,
} from "@deck.gl-community/editable-layers";
import {
  Color,
  DeckGL,
  DeckGLRef,
  MapViewState,
  PickingInfo,
  ScatterplotLayer,
} from "deck.gl";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Map, MapRef } from "react-map-gl";
import { INITIAL_VIEW_STATE } from "@/lib/constants";
import LatLngDisplay from "@/components/widgets/InfoBar";
import ImageUpload from "@/components/widgets/imageUpload";
import OperationContainer from "@/components/widgets/ops";
import ky from "ky";
import dynamic from "next/dynamic";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
const ESearchBox = dynamic(() => import("@/components/widgets/searchBox"), {
  ssr: false,
});
import "mapbox-gl/dist/mapbox-gl.css";
import { Msg, ingestStream } from "@/lib/rpc";
import { useRouter } from "next/navigation";
import { useSift } from "@/app/sift/siftStore";
import { columnHelper } from "../sift/table";
import { TableItemsFromCoord, formatValue, getStats, zVal } from "@/lib/utils";
import { NumberPill } from "@/components/pill";

const selectedFeatureIndexes: number[] = [];

export default function StreetView() {
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState(false);
  const [image, setImage] = useState<string | null>(null);
  const [sampling, setSampling] = useState(false);
  const [locating, setLocating] = useState(false);
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
  const [distKm, setDistKm] = useState(0.05);
  const deckRef = useRef<DeckGLRef>(null);
  const [featCollection, setFeatCollection] = useState<FeatureCollection>({
    type: "FeatureCollection",
    features: [],
  });
  const [cursorCoords, setCursorCoords] = useState<Point>({
    lat: 0,
    lon: 0,
    aux: null,
  });
  const [sampled, setSampled] = useState<Coords | null>(null);
  const [located, setLocated] = useState<Coords | null>(null);
  const { setCols, addItems, setTargetImage } = useSift();
  const [topN, setTopN] = useState(20);
  const router = useRouter();
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
  const viewedLocated = useMemo(() => {
    return located
      ? locating
        ? located.coords
        : located.coords.slice(0, topN)
      : null;
  }, [located, topN, locating]);

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

  const sampledLayer = new ScatterplotLayer({
    id: "results-layer",
    data: sampled?.coords,
    getPosition: (d) => [d.lon, d.lat],
    getRadius: (d) => 1,
    getFillColor: (d) => [255, 140, 0],
    pickable: true,
    radiusScale: 1,
    radiusMinPixels: 2,
    radiusMaxPixels: 100,
    visible: !!sampled && !located,
  });

  const locateResultsLayer = new ScatterplotLayer<Point>({
    id: "locate-results-layer",
    data: viewedLocated,
    getPosition: (d) => [d.lon, d.lat],
    getRadius: (d) => 1,
    getFillColor: (d) =>
      [Math.floor(255 * Math.sqrt(d.aux.max_sim)), 140, 0] as Color,
    pickable: true,
    radiusMinPixels: 2,
    radiusMaxPixels: 100,
  });

  const getTooltip = useCallback(({ object }: PickingInfo<Point>) => {
    if (!object?.lat) return null;
    return object
      ? `Coordinates: ${object.lat.toFixed(4)}, ${object.lon.toFixed(4)}
      Click to copy full coordinates
      ${object.aux.max_sim ? `Similarity: ${object.aux.max_sim}` : ""}`
      : null;
  }, []);

  const onSample = () => {
    setSampling(true);
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
    ky.post(`${API_URL}/streetview/sample`, {
      timeout: false,
      json: {
        bounds,
        dist_km: distKm,
      },
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        setSampled(data as Coords);
        setSampling(false);
      });
  };

  const onLocate = async () => {
    const payload = {
      image_url: image,
      coords: { coords: sampled!.coords },
    };

    const response = await ky.post(`${API_URL}/streetview/locate/streaming`, {
      timeout: false,
      json: payload,
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to start locate: ${response.statusText}`);
    }
    if (!response.body) {
      throw new Error("Failed to start locate: API returned no response body");
    }

    for await (const msg of ingestStream(response)) {
      console.log("got message! ", msg);
      switch (msg.type) {
        case "Coords":
          setLocated((loc) => {
            if (loc?.coords?.length) {
              console.log("Old coords:", loc.coords);
              let new_cords = [...loc.coords, ...msg.coords].sort(
                (a, b) => b.aux.max_sim - a.aux.max_sim
              );
              console.log("New coords (extension):", new_cords);
              return {
                coords: new_cords,
              };
            }
            console.log("New coords:", msg.coords);
            return msg;
          });
          break;
        case "ProgressUpdate":
          console.log(msg);
          break;
        default:
          break;
      }
    }
  };

  const locateWrapper = async () => {
    setLocating(true);
    try {
      await onLocate();
    } catch (e) {
      console.log(e);
    } finally {
      setLocating(false);
    }
  };

  const activeLayers = useMemo(() => {
    if (located) {
      return [layer, locateResultsLayer];
    } else {
      return [layer, sampledLayer];
    }
  }, [located, sampled, layer, locateResultsLayer, sampledLayer]);

  return (
    <div>
      <div
        className={`absolute w-full h-full ${
          selecting ? "cursor-crosshair" : ""
        }`}
      >
        <DeckGL
          initialViewState={viewState}
          controller
          layers={activeLayers}
          getTooltip={getTooltip}
          ref={deckRef}
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
          <h3>Streetview Geolocalization</h3>
          Select an area, and this will iterate through streetview images within
          that area to find the best match for your image.
        </article>
        <div className="flex flex-col gap-2">
          <ImageUpload
            onSetImage={setImage}
            image={image}
            className="border-stone-400"
          />
          <Label htmlFor="dist-slider">Sample Density: {distKm * 1000} m</Label>
          <Slider
            id="dist-slider"
            value={[distKm]}
            onValueChange={(v) => setDistKm(v[0])}
            min={0.01}
            max={0.1}
            step={0.001}
          />
          {selecting || selected ? (
            <div className="flex flex-row gap-2">
              <Button
                disabled={featCollection.features.length === 0 || sampling}
                onClick={() => {
                  console.log(featCollection);
                  onSample();
                }}
              >
                {sampling ? "Fetching..." : "Fetch Streetviews"}
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
                Cancel
              </Button>
            </div>
          ) : (
            <Button onClick={() => setSelecting(true)}>
              Select Search Range
            </Button>
          )}
          <Button onClick={locateWrapper} disabled={!sampled || locating}>
            {locating ? "Locating..." : "Run Geolocalizaion"}
          </Button>
          <Label htmlFor="topn-slider">Top {topN} Results</Label>
          <Slider
            id="topn-slider"
            value={[topN]}
            onValueChange={(v) => setTopN(v[0])}
            min={1}
            max={sampled?.coords.length || 50}
            step={1}
          />
          <Button
            onClick={() => {
              const stats = getStats(located!.coords.map((c) => c.aux.max_sim));
              setTargetImage(image!);
              setCols((cols) => [
                ...cols,
                {
                  type: "NumericalCol",
                  accessor: "max_sim",
                  header: "Streetview Similarity",
                  ...stats,
                },
              ]);
              addItems(TableItemsFromCoord(located!));
              router.push("/sift");
            }}
            disabled={!located || !image}
          >
            Sift Through Results
          </Button>
        </div>
      </OperationContainer>
      <ESearchBox setViewState={setViewState} dglref={deckRef} />
      <LatLngDisplay cursorCoords={cursorCoords} />
    </div>
  );
}
