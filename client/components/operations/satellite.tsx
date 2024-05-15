import { Controller, DeckGL, MapController, MapViewState } from "deck.gl";
import { MAPBOX_TOKEN } from "@/lib/constants";
import { Map, MapRef, ViewState } from "react-map-gl";
import OperationContainer from "./ops";
import { Button } from "@/components/ui/button";
import { INITIAL_VIEW_STATE } from "../map";
import { useMemo, useRef, useState } from "react";
import { PathLayer } from "@deck.gl/layers";
import { Bounds, Point } from "@/lib/geo";
import { MjolnirEvent } from "mjolnir.js";

export default function Satellite() {
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState(false);
  const [hiBound, setHiBound] = useState<Point | null>(null);
  const [loBound, setLoBound] = useState<Point | null>(null);
  const [cursorLngLat, setCursorLngLat] = useState<Point | null>(null);
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
  const mapRef = useRef<MapRef>(null);
  let bound = useMemo(() => {
    if (!hiBound) return null;
    const lo = loBound ? loBound : cursorLngLat;
    return { hi: hiBound, lo };
  }, [hiBound, loBound, cursorLngLat]);

  const layer = new PathLayer<Bounds | null>({
    id: "PathLayer",
    data: [bound],
    getPath: (d) => {
      if (!d) return [];
      return [
        [d.hi.lon, d.hi.lat],
        [d.hi.lon, d.lo.lat],
        [d.lo.lon, d.lo.lat],
        [d.lo.lon, d.hi.lat],
        [d.hi.lon, d.hi.lat],
      ];
    },
    getColor: () => [0, 0, 255],
    getWidth: () => 2,
    pickable: true,
  });

  class EKController extends MapController {
    constructor(props: any) {
      super(props);
      this.events = ["pointermove"];
    }

    handleEvent(event: MjolnirEvent) {
      if (event.type === "pointermove") {
        // const lnglat = this.controllerState._unproject([
        //   event.center.x,
        //   event.center.y,
        // ]) as [number, number];
        // setCursorLngLat({
        //   lon: lnglat[0],
        //   lat: lnglat[1],
        //   aux: null,
        // });
      } else if (event.type == "pointerdown" && selecting) {
        setHiBound(cursorLngLat);
        return true;
      } else if (event.type == "pointerup" && selecting) {
        setLoBound(cursorLngLat);
      }
      return super.handleEvent(event);
    }
  }

  return (
    <div>
      <div
        className={`absolute w-full h-full ${
          selecting ? "cursor-crosshair" : ""
        }`}
      >
        <DeckGL
          initialViewState={INITIAL_VIEW_STATE}
          controller={{
            type: EKController,
            dragMode: "pan",
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
