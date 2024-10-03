"use client";

import { Button } from "@/components/ui/button";
import ImageUpload from "@/components/widgets/imageUpload";
import OperationContainer from "@/components/widgets/ops";
import { useAPIClient } from "@/lib/api-client/api";
import {
  API_URL,
  DEFAULT_MAP_STYLE,
  INITIAL_VIEW_STATE,
  MAPBOX_TOKEN,
} from "@/lib/constants";
import { Coords, Point, getbbox } from "@/lib/geo";
import {
  EditableGeoJsonLayer,
  DrawPointMode,
  ViewMode,
  FeatureCollection,
} from "@deck.gl-community/editable-layers";
import { feature } from "@turf/turf";
import {
  DeckGL,
  DeckGLRef,
  FlyToInterpolator,
  GeoJsonLayer,
  HeatmapLayer,
  Layer,
  MapViewState,
  PickingInfo,
  WebMercatorViewport,
} from "deck.gl";
import { Copy, Loader2 } from "lucide-react";
import { useCallback, useMemo, useState, useRef } from "react";
import { AttributionControl, Map } from "react-map-gl/maplibre";
import { toast } from "sonner";
import dynamic from "next/dynamic";
import InfoBar from "@/components/widgets/InfoBar";
const ESearchBox = dynamic(() => import("@/components/widgets/searchBox"), {
  ssr: false,
});

export default function Calibrate() {
  const [image, setImage] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [calibrated, setCalibrated] = useState<Point | null>(null);
  const [featCollection, setFeatCollection] = useState<FeatureCollection>({
    type: "FeatureCollection",
    features: [],
  });
  const getClient = useAPIClient("api");
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
  const deckRef = useRef<DeckGLRef>(null);

  const [cursorCoords, setCursorCoords] = useState<Point>({
    lat: 0,
    lon: 0,
    aux: null,
  });

  const onInference = async () => {
    setIsRunning(true);
    setCalibrated(null);
    if (!image || !curPoint) {
      toast.error("Please upload an image and set a point");
      setIsRunning(false);
      return;
    }
    let apiClient = await getClient();
    const { data, error } = await apiClient.POST("/orienternet/locate", {
      body: {
        image_url: image,
        location_prior: curPoint,
      },
    });
    if (error) {
      toast.error(error.detail);
      setIsRunning(false);
      return;
    }
    setCalibrated(data);
    setIsRunning(false);
  };

  const curPoint: Point | null = useMemo(() => {
    console.log("featCollection", featCollection);
    if (!featCollection || featCollection.features.length === 0) {
      return null;
    }
    const feat = featCollection.features[0];
    console.log("feat", feat);
    return {
      lat: feat.geometry.coordinates[1] as number,
      lon: feat.geometry.coordinates[0] as number,
      aux: null,
    };
  }, [featCollection]);

  const selectLayer = new EditableGeoJsonLayer({
    id: "select-layer",
    mode: DrawPointMode,
    data: featCollection,
    onEdit: ({ updatedData }) => {
      if (updatedData.features.length > 1) {
        const lastFeature =
          updatedData.features[updatedData.features.length - 1];
        setFeatCollection({
          type: "FeatureCollection",
          features: [lastFeature],
        });
      } else {
        setFeatCollection(updatedData);
      }
    },
    getFillColor: [0, 0, 0, 255],
    getRadius: 10,
  });

  const calibratedFeatCollection: FeatureCollection = useMemo(() => {
    return {
      type: "FeatureCollection",
      features: calibrated
        ? [
            {
              type: "Feature",
              geometry: {
                type: "Point",
                coordinates: [calibrated.lon, calibrated.lat],
              },
              properties: {
                id: "calibrated",
              },
            },
          ]
        : [],
    };
  }, [calibrated]);

  const calibratedLayer = new GeoJsonLayer({
    id: "calibrated-layer",
    data: calibratedFeatCollection as any,
    getFillColor: [255, 0, 0, 255],
    getRadius: 10,
  });

  const trackingVm = useMemo(() => {
    let vm = ViewMode;
    vm.prototype.handlePointerMove = ({ mapCoords }) => {
      setCursorCoords({
        lon: mapCoords[0],
        lat: mapCoords[1],
        aux: null,
      });
    };
    return vm;
  }, []);

  const trackingLayer = new EditableGeoJsonLayer({
    id: "tracking-layer",
    data: {
      type: "FeatureCollection",
      features: [],
    },
    mode: trackingVm,
  });

  return (
    <div className="w-full h-full relative p-2 overflow-hidden">
      <DeckGL
        ref={deckRef}
        initialViewState={viewState}
        controller
        layers={[selectLayer, calibratedLayer, trackingLayer]}
        getCursor={(st) => (st.isDragging ? "grabbing" : "crosshair")}
      >
        <Map mapStyle={DEFAULT_MAP_STYLE} attributionControl={false}>
          <AttributionControl position="bottom-left" />
        </Map>
      </DeckGL>
      <OperationContainer className="w-64 flex flex-col gap-2">
        <article className="prose prose-sm leading-5">
          <h3>Orienternet Calibration</h3>
          <a href="https://github.com/facebookresearch/OrienterNet">
            Orienternet
          </a>{" "}
          takes an image and a location prior and returns a calibrated
          coordinate.
        </article>
        <ImageUpload
          onSetImage={(img) => {
            setImage(img);
          }}
          image={image}
        />
        <div className="flex flex-col items-center">
          {curPoint ? (
            <div className="w-full p-2 bg-gray-100 rounded-md mb-2">
              <p className="text-sm font-semibold">Selected Location:</p>
              <p className="text-xs">
                Lat: {curPoint.lat.toFixed(6)}, Lon: {curPoint.lon.toFixed(6)}
              </p>
            </div>
          ) : (
            <span className="text-sm text-gray-500">
              Click on map to select location
            </span>
          )}
          {calibrated && (
            <div className="w-full p-2 bg-green-100 rounded-md mb-2">
              <p className="text-sm font-semibold">Calibrated Location:</p>
              <p className="text-xs">
                Lat: {calibrated.lat.toFixed(6)}, Lon:{" "}
                {calibrated.lon.toFixed(6)}
              </p>
              <Button
                variant="outline"
                size="sm"
                className="mt-2 w-full"
                onClick={() =>
                  navigator.clipboard.writeText(
                    `${calibrated.lat.toFixed(6)},${calibrated.lon.toFixed(6)}`
                  )
                }
              >
                <Copy className="mr-2 h-4 w-4" />
                Copy Coordinates
              </Button>
            </div>
          )}
          <Button
            className={`mt-3 w-full`}
            disabled={!image || !curPoint || isRunning}
            onClick={onInference}
          >
            {isRunning ? <Loader2 className="animate-spin mr-2" /> : null}
            {isRunning ? "Calibrating..." : "Calibrate"}
          </Button>
        </div>
      </OperationContainer>
      <ESearchBox setViewState={setViewState} dglref={deckRef} />
      <InfoBar
        className="bottom-12"
        showShortcuts
        cursorCoords={cursorCoords}
      />
    </div>
  );
}
