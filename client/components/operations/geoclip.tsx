"use client";
import { Button } from "@/components/ui/button";
import { API_URL, MAPBOX_TOKEN } from "@/lib/constants";
import { Coords, Point, getbbox } from "@/lib/geo";
import {
  DeckGL,
  FlyToInterpolator,
  HeatmapLayer,
  MapViewState,
  PickingInfo,
  WebMercatorViewport,
} from "deck.gl";
import { Loader2 } from "lucide-react";
import { useCallback, useState } from "react";
import { Map } from "react-map-gl";
import { INITIAL_VIEW_STATE } from "@/lib/constants";
import ImageUpload from "@/components/widgets/imageUpload";
import OperationContainer from "@/components/widgets/ops";
import "mapbox-gl/dist/mapbox-gl.css";
import { getHeaders } from "@/lib/supabase/client";

export default function GeoCLIP() {
  const [image, setImage] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [predictions, setPredictions] = useState<Coords | null>(null);
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);

  const onInference = async () => {
    setIsRunning(true);
    setPredictions(null);
    fetch(`${API_URL}/geoclip`, {
      method: "POST",
      body: JSON.stringify({
        image_url: image,
        top_k: 100,
      }),
      ...(await getHeaders()),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
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
        const bounds = getbbox(adjusted_data);
        console.log(adjusted_data);
        console.log(bounds);
        const { longitude, latitude, zoom } = vp.fitBounds(
          [
            [bounds.lo.lat, bounds.lo.lon],
            [bounds.hi.lat, bounds.hi.lon],
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
    getPosition: (d) => [d.lat, d.lon],
    getWeight: (d) => d.aux.conf,
    pickable: true,
    radiusPixels: 25,
  });

  const getTooltip = useCallback(({ object }: PickingInfo<Point>) => {
    return object
      ? `Coordinates: ${object.lat.toFixed(4)}, ${object.lon.toFixed(4)}
      Confidence: ${object.aux.conf.toFixed(2)}
      Click to copy full coordinates`
      : null;
  }, []);

  return (
    <DeckGL
      initialViewState={viewState}
      controller
      layers={[layer]}
      getTooltip={getTooltip}
    >
      <Map
        mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
        mapboxAccessToken={MAPBOX_TOKEN}
      ></Map>
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
          onUploadBegin={() => {
            fetch(`${API_URL}/geoclip/poke`);
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
        </div>
      </OperationContainer>
    </DeckGL>
  );
}
