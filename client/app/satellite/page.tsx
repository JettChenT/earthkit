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
import { AttributionControl, Map, MapRef } from "react-map-gl";
import { INITIAL_VIEW_STATE } from "@/lib/constants";
import InfoBar from "@/components/widgets/InfoBar";
import ImageUpload from "@/components/widgets/imageUpload";
import OperationContainer from "@/components/widgets/ops";
import dynamic from "next/dynamic";
const ESearchBox = dynamic(() => import("@/components/widgets/searchBox"), {
  ssr: false,
});
import "mapbox-gl/dist/mapbox-gl.css";
import { useSWRConfig } from "swr";
import { useKy } from "@/lib/api-client/api";
import { toast } from "sonner";
import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import * as turf from "@turf/turf";

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

  const [isAreaExceeded, setIsAreaExceeded] = useState(false);
  const getBounds = useCallback(
    (collection: FeatureCollection | null = null) => {
      let feats = collection ?? featCollection;
      const bounds: Bounds = {
        lo: {
          // @ts-ignore
          lat: feats.features[0].geometry.coordinates[0][0][1],
          // @ts-ignore
          lon: feats.features[0].geometry.coordinates[0][0][0],
          aux: null,
        },
        hi: {
          // @ts-ignore
          lat: feats.features[0].geometry.coordinates[0][2][1],
          // @ts-ignore
          lon: feats.features[0].geometry.coordinates[0][2][0],
          aux: null,
        },
      };
      return bounds;
    },
    [featCollection]
  );

  const layer = new EditableGeoJsonLayer({
    id: "geojson-layer",
    data: featCollection,
    mode: viewMode,
    selectedFeatureIndexes,
    getLineColor: (d) => {
      if (isAreaExceeded) return [255, 0, 0, 90];
      return [0, 0, 0, 90];
    },
    getFillColor: (d) => {
      if (isAreaExceeded) return [255, 0, 0, 50];
      return [0, 0, 0, 90];
    },
    onEdit: ({ updatedData }) => {
      let newFeatures: FeatureCollection = {
        type: "FeatureCollection",
        features: updatedData.features.slice(-1),
      };
      setFeatCollection(newFeatures);
      let area = turf.area(newFeatures as any);
      setIsAreaExceeded(area > 2_000_000);
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

  const { mutate } = useSWRConfig();
  const getKyInst = useKy();

  const onLocate = async () => {
    setLocating(true);
    setSelected(true);
    setSelecting(false);
    console.log(featCollection);
    const bounds: Bounds = getBounds();
    const ky = await getKyInst();
    ky.post(`satellite/locate`, {
      json: {
        bounds,
        image_url: image,
      },
    })
      .then((res) => res.json())
      .then((data) => {
        setResults(data as Coords);
        setLocating(false);
        mutate("/api/usage");
      })
      .catch((err) => {
        toast.error(err.message);
        setLocating(false);
      });
  };

  const deckRef = useRef<DeckGLRef>(null);

  return (
    <div className="relative h-screen w-full overflow-hidden">
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
            attributionControl={false}
          >
            <AttributionControl position="bottom-left" />
          </Map>
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
          <div>
            {isAreaExceeded && (
              <Alert variant="destructive" className="bg-white bg-opacity-75">
                <AlertCircle className="size-4" />
                <AlertTitle>Area exceeded</AlertTitle>
                <AlertDescription>
                  Please select a smaller area.
                </AlertDescription>
              </Alert>
            )}
          </div>
          {selecting || selected ? (
            <div className="flex flex-row gap-2">
              <Button
                disabled={
                  featCollection.features.length === 0 ||
                  locating ||
                  isAreaExceeded
                }
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
      <InfoBar cursorCoords={cursorCoords} />
      <ESearchBox setViewState={setViewState} dglref={deckRef} />
    </div>
  );
}
