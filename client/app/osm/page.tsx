"use client";

import { Chatbox } from "@/components/chatbox";
import { useChat } from "ai/react";

import { ChatMessages } from "@/components/chat-messages";
import ky from "ky";
import DeckGL, { GeoJsonLayer } from "deck.gl";
import { Map } from "react-map-gl";
import { INITIAL_VIEW_STATE, MAPBOX_TOKEN } from "@/lib/constants";
import osmtogeojson from "osmtogeojson";
import { useState } from "react";
import "mapbox-gl/dist/mapbox-gl.css";

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

export default function OSM() {
  const [geojsonData, setGeojsonData] = useState(null);
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "api/osmnl",
    async onFinish({ content }) {
      const osm_codeblock = content
        .split("```overpassql\n")
        .at(-1)
        ?.split("\n```")
        .at(0);
      if (!osm_codeblock) return;
      console.log(osm_codeblock);
      const result = await ky
        .post(OVERPASS_URL, {
          body: "data=" + encodeURIComponent(osm_codeblock),
          timeout: false,
        })
        .json()
        .catch((e) => {
          console.log(e);
        });
      console.log(`result: ${result}`);
      const geojson = osmtogeojson(result);
      console.log(geojson);
      setGeojsonData(geojson);
    },
  });

  const layers = [
    geojsonData &&
      new GeoJsonLayer({
        id: "geojson-layer",
        data: geojsonData,
        pickable: true,
        stroked: false,
        filled: true,
        extruded: true,
        pointType: "circle",
        getFillColor: [255, 165, 0, 200], // Changed to a more orangy color
        getLineColor: [255, 255, 255],
        getPointRadius: 10,
        pointRadiusMinPixels: 3,
        getLineWidth: 1,
        getElevation: 30,
      }),
  ];

  return (
    <div className="w-full h-screen flex flex-row gap-3">
      <div className="flex-1 flex flex-col overflow-hidden justify-start">
        <ChatMessages messages={messages} />
        <Chatbox
          handleSubmit={handleSubmit}
          handleInputChange={handleInputChange}
          input={input}
        />
      </div>
      <div className="flex-1 p-3">
        <div className="h-full relative">
          <DeckGL
            initialViewState={INITIAL_VIEW_STATE}
            controller
            layers={layers}
          >
            <Map
              mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
              mapboxAccessToken={MAPBOX_TOKEN}
            ></Map>
          </DeckGL>
        </div>
      </div>
    </div>
  );
}
