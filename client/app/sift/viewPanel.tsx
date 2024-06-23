"use client";

import EmbedMap from "@/components/embed-map";
import MapMarker from "@/components/icons/mapMarker";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DEFAULT_MAP_STYLE, INITIAL_VIEW_STATE } from "@/lib/constants";
import bbox from "@turf/bbox";
import { FlyToInterpolator } from "deck.gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useMemo, useRef, useState } from "react";
import Map, { MapRef, Marker } from "react-map-gl/maplibre";
import { createGeoJson } from "./inout";
import { ViewPanelType, useSift } from "./siftStore";

export default function ViewPanel() {
  const { viewPanelState, setViewPanelState } = useSift();
  const currentItem = useSift((state) => state.getSelected());
  return (
    <Tabs
      value={viewPanelState}
      onValueChange={(value) => setViewPanelState(value as ViewPanelType)}
      className="h-full flex flex-col gap-1 mt-4"
    >
      <div>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger disabled={!currentItem} value="streetview">
            Street View
          </TabsTrigger>
          <TabsTrigger disabled={!currentItem} value="map">
            Map
          </TabsTrigger>
          <TabsTrigger disabled={!currentItem} value="satellite">
            Satellite
          </TabsTrigger>
        </TabsList>
      </div>
      <div className="grow">
        {viewPanelState == "overview" ? (
          <MapOverview />
        ) : (
          currentItem && (
            <EmbedMap
              panoId={currentItem.aux?.panoId}
              coord={currentItem.coord}
              viewType={viewPanelState}
            />
          )
        )}
      </div>
    </Tabs>
  );
}

function MapOverview() {
  let { items, cols, idx } = useSift();
  let mapRef = useRef<MapRef>(null);
  let [loaded, setLoaded] = useState(false);

  const viewState = useMemo(() => {
    console.debug("viewState: dependencies changed", items, cols, mapRef);
    if (!items.length) return INITIAL_VIEW_STATE;
    let gjson = createGeoJson({ items, cols });
    let [minLng, minLat, maxLng, maxLat] = bbox(gjson);
    if (!mapRef.current) return INITIAL_VIEW_STATE;
    let { longitude, latitude, zoom } = mapRef.current.fitBounds(
      [
        [minLng, minLat],
        [maxLng, maxLat],
      ],
      { padding: 100 }
    );
    console.debug("reached the end of update");
    return {
      longitude,
      latitude,
      zoom: Math.max(zoom - 1, 2),
      transitionInterpolator: new FlyToInterpolator({ speed: 2 }),
      transitionDuration: "auto",
    };
  }, [items, cols, mapRef, mapRef.current, loaded]);

  useEffect(() => {
    if (!mapRef.current) return;
    let item = items.at(idx);
    if (!item) return;
    if (
      !mapRef.current.getBounds().contains([item.coord.lon, item.coord.lat])
    ) {
      mapRef.current.flyTo({
        center: [item.coord.lon, item.coord.lat],
      });
    }
  }, [idx, items]);

  let markers = useMemo(() => {
    return items.map((item, itemIdx) => (
      <Marker
        key={itemIdx}
        latitude={item.coord.lat}
        longitude={item.coord.lon}
        style={{
          zIndex: itemIdx === idx ? 50 : 0,
        }}
      >
        <MapMarker fill={itemIdx === idx ? "red" : "#3FB1CE"} />
      </Marker>
    ));
  }, [items, idx]);

  return (
    <div className="w-full h-full relative rounded-md p-2 pb-6">
      <Map
        ref={mapRef}
        onLoad={() => setLoaded(true)}
        initialViewState={viewState}
        mapStyle={DEFAULT_MAP_STYLE}
      >
        <div className="absolute top-5 right-5 font-mono p-2 bg-muted/50 rounded-md">
          {items.length} items
        </div>
        {markers}
      </Map>
    </div>
  );
}
