import { FileUploader } from "react-drag-drop-files";
import { useCallback, useState } from "react";
import { useStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { API_URL, MAPBOX_TOKEN } from "@/lib/constants";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import {
  DeckGL,
  FlyToInterpolator,
  MapViewState,
  PickingInfo,
  ScatterplotLayer,
  WebMercatorViewport,
} from "deck.gl";
import { INITIAL_VIEW_STATE } from "../map";
import { Map } from "react-map-gl";
import OperationContainer from "./ops";
import { Coords, Point, getbbox } from "@/lib/geo";

const fileTypes = ["JPG", "PNG", "GIF"];

export default function GeoCLIP() {
  const [image, setImage] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [predictions, setPredictions] = useState<Coords | null>(null);
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);

  const onUpload = (file: File) => {
    console.log("on drop!");
    console.log(file);
    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function () {
      setImage(reader.result as string);
    };
  };

  const onInference = () => {
    setIsRunning(true);
    setPredictions(null);
    fetch(`${API_URL}/geoclip`, {
      method: "POST",
      body: JSON.stringify({
        image_url: image,
      }),
      headers: {
        "Content-Type": "application/json",
      },
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
          zoom,
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

  const layer = new ScatterplotLayer<Point>({
    id: "geoclip_pred",
    data: predictions?.coords,
    getPosition: (d) => [d.lat, d.lon],
    getRadius: (d) => 0.1,
    getFillColor: (d) => [d.aux.conf * 255, 0, 0],
    pickable: true,
    radiusScale: 1,
    radiusMinPixels: 5,
    radiusMaxPixels: 100,
    onClick: (info: PickingInfo<Point>) => {
      if (!info.object) return;
      const { lat, lon } = info.object;
      toast(`Copied ${lat}, ${lon} to clipboard`);
      navigator.clipboard.writeText(`${lat}, ${lon}`);
    },
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
          <h3 className="font-bold">GeoCLIP Geoestimation</h3>
          <a
            className="text-primary"
            href="https://github.com/VicenteVivan/geo-clip"
          >
            GeoCLIP
          </a>{" "}
          predicts the location of an image based on its visual features.
        </article>
        {image ? (
          <img className="rounded-md" src={image} />
        ) : (
          <FileUploader handleChange={onUpload} name="file" types={fileTypes}>
            <div className="w-full h-32 bg-slate-300 bg-opacity-20 rounded-md flex items-center justify-center hover:bg-slate-300 hover:bg-opacity-30 border-dashed border-2 border-slate-200 hover:cursor-pointer">
              <div className="text-lg font-bold">Import Image</div>
            </div>
          </FileUploader>
        )}
        <div className="flex flex-col items-center">
          <Button
            className={`mt-3 w-full`}
            disabled={!image || isRunning}
            onClick={onInference}
          >
            {isRunning ? <Loader2 className="animate-spin mr-2" /> : null}
            {isRunning ? "Predicting..." : "Predict"}
          </Button>
          <Button
            className={`mt-3 w-full`}
            variant="secondary"
            onClick={onCancel}
            disabled={!image}
          >
            Cancel
          </Button>
        </div>
      </OperationContainer>
    </DeckGL>
  );
}
