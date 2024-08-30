"use client";
import { Button } from "@/components/ui/button";
import { API_URL, DEFAULT_MAP_STYLE, MAPBOX_TOKEN } from "@/lib/constants";
import { Coords, Point, getbbox } from "@/lib/geo";
import {
  DeckGL,
  FlyToInterpolator,
  HeatmapLayer,
  Layer,
  MapViewState,
  PickingInfo,
  WebMercatorViewport,
} from "deck.gl";
import { CopyIcon, Loader2 } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { AttributionControl, Map } from "react-map-gl/maplibre";
import { INITIAL_VIEW_STATE } from "@/lib/constants";
import ImageUpload from "@/components/widgets/imageUpload";
import OperationContainer from "@/components/widgets/ops";
import { useAPIClient, useKy } from "@/lib/api-client/api";
import { toast } from "sonner";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  EditableGeoJsonLayer,
  ViewMode,
} from "@deck.gl-community/editable-layers";
import InfoBar from "../widgets/InfoBar";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuShortcut,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { useHotkeys } from "react-hotkeys-hook";
import Kbd, { MetaKey } from "../keyboard";
import { parseGeoJsonImport } from "@/app/sift/inout";
import { FeatureCollection, Point as GeoJSONPoint } from "geojson";
import { useSift } from "@/app/sift/siftStore";
import { useRouter } from "next/navigation";
import { TableItemsFromCoord, downloadContent, getStats } from "@/lib/utils";

export default function GeoCLIP() {
  const [image, setImage] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [predictions, setPredictions] = useState<Coords | null>(null);
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
  const getKyInst = useKy();
  const getClient = useAPIClient("ai");

  const onInference = async () => {
    setIsRunning(true);
    setPredictions(null);
    let apiClient = await getClient();
    if (!image) {
      toast.error("Please upload an image");
      setIsRunning(false);
      return;
    }
    const { data, error } = await apiClient.POST("/geoclip", {
      body: {
        image_url: image,
        top_k: 100,
      },
    });
    if (error) {
      toast.error(error.detail);
      setIsRunning(false);
      return;
    }
    const max_conf = Math.max(...data.map((d: any) => d.aux.pred));
    const adjusted_data: Coords = {
      coords: data.map((d: any) => {
        d.aux.conf = Math.sqrt(d.aux.pred / max_conf);
        return d;
      }),
    };
    setPredictions(adjusted_data);
    setIsRunning(false);
    const vp = layer.context.viewport as WebMercatorViewport;
    const bounds = getbbox(adjusted_data.coords);
    console.log(adjusted_data);
    console.log(bounds);
    const { longitude, latitude, zoom } = vp.fitBounds(
      [
        [bounds.lo.lon, bounds.lo.lat],
        [bounds.hi.lon, bounds.hi.lat],
      ],
      { padding: 100 }
    );
    setViewState({
      longitude,
      latitude,
      zoom: Math.max(zoom - 2, 2),
      transitionInterpolator: new FlyToInterpolator({ speed: 2 }),
      transitionDuration: "auto",
    });
  };

  const onCancel = () => {
    setIsRunning(false);
    setImage(null);
    setPredictions(null);
    setViewState({
      ...INITIAL_VIEW_STATE,
      transitionInterpolator: new FlyToInterpolator({ speed: 4 }),
      transitionDuration: "auto",
    });
  };

  const layer = new HeatmapLayer<Point>({
    id: "geoclip_pred",
    data: predictions?.coords,
    getPosition: (d) => [d.lon, d.lat],
    getWeight: (d) => d.aux.conf,
    pickable: true,
    radiusPixels: 25,
  });

  // HACK: uses editable layers when it's probably not necessary
  // TODO: make this a useCursorCoords thing
  const [cursorCoords, setCursorCoords] = useState<Point>({
    lat: 0,
    lon: 0,
    aux: null,
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

  const getTooltip = useCallback(({ object }: PickingInfo<Point>) => {
    return object
      ? `Coordinates: ${object.lat.toFixed(4)}, ${object.lon.toFixed(4)}
      Confidence: ${object.aux.conf.toFixed(2)}
      Click to copy full coordinates`
      : null;
  }, []);

  const copyCoords = useCallback(() => {
    navigator.clipboard.writeText(`${cursorCoords.lat}, ${cursorCoords.lon}`);
    toast.success("Copied coordinates to clipboard");
  }, [cursorCoords]);

  const getFeats = useCallback(() => {
    if (!predictions) return;
    const feats: FeatureCollection = {
      type: "FeatureCollection",
      features: predictions.coords.map((c) => ({
        type: "Feature",
        properties: { geoclip_score: c.aux.conf },
        geometry: { type: "Point", coordinates: [c.lon, c.lat] },
      })),
    };
    return feats;
  }, [predictions]);

  const { addItems, setCols, setTargetImage } = useSift();
  const router = useRouter();

  const toSift = useCallback(() => {
    if (!predictions) return;
    const res = TableItemsFromCoord(predictions);
    const stats = getStats(predictions.coords.map((c) => c.aux.conf));
    setCols((cols) => [
      ...cols,
      {
        type: "NumericalCol",
        accessor: "conf",
        header: "Geoclip Score",
        ...stats,
      },
    ]);
    addItems(res);
    router.push("/sift");
    console.debug("current image", image);
    setTargetImage(image);
  }, [getFeats]);

  const toExport = useCallback(() => {
    const feats = getFeats();
    downloadContent(JSON.stringify(feats), "geojson");
  }, [getFeats]);

  useHotkeys("Meta+C", copyCoords);

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div className="w-full h-full relative p-2 overflow-hidden">
          <DeckGL
            initialViewState={viewState}
            controller
            layers={[layer, trackingLayer]}
            getTooltip={getTooltip}
            getCursor={(st) => (st.isDragging ? "grabbing" : "crosshair")}
          >
            <Map mapStyle={DEFAULT_MAP_STYLE} attributionControl={false}>
              <AttributionControl position="bottom-left" />
            </Map>
            <OperationContainer className="w-64">
              <article className="prose prose-sm leading-5 mb-2">
                <h3>GeoCLIP Geoestimation</h3>
                <a
                  className="text-primary"
                  href="https://github.com/VicenteVivan/geo-clip"
                >
                  GeoCLIP
                </a>{" "}
                predicts the location of an image based on its visual features.
              </article>
              <ImageUpload
                onSetImage={(img) => {
                  setImage(img);
                }}
                image={image}
              />
              <div className="flex flex-col items-center">
                <Button
                  className={`mt-3 w-full`}
                  disabled={!image || isRunning}
                  onClick={onInference}
                >
                  {isRunning ? <Loader2 className="animate-spin mr-2" /> : null}
                  {isRunning ? "Predicting..." : "Predict"}
                </Button>
                {image && (
                  <Button
                    className={`mt-3 w-full`}
                    variant="secondary"
                    onClick={onCancel}
                  >
                    Cancel
                  </Button>
                )}
                {predictions && (
                  <div className="mt-3 w-full px-1">
                    <div className="flex flex-col gap-2 rounded-md border p-2">
                      <h2 className="text-md font-bold">More Actions</h2>
                      <Button variant="outline" onClick={toSift}>
                        Sift Top 100 Coords
                      </Button>
                      <Button variant="outline" onClick={toExport}>
                        Export Top 100 Coords
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </OperationContainer>
          </DeckGL>
          <InfoBar
            className="bottom-12"
            showShortcuts
            cursorCoords={cursorCoords}
          />
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent className="w-56">
        <ContextMenuItem onSelect={copyCoords}>
          <CopyIcon className="w-4 h-4 mr-2" /> Copy Coordinates
          <ContextMenuShortcut>
            <MetaKey noWrap />
            +C
          </ContextMenuShortcut>
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
}
