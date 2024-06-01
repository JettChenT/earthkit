"use client";

import { Chatbox } from "@/components/chatbox";
import { ChatMessages } from "@/components/chat-messages";
import ky from "ky";
import DeckGL, { GeoJsonLayer } from "deck.gl";
import { Map } from "react-map-gl";
import { INITIAL_VIEW_STATE, MAPBOX_TOKEN } from "@/lib/constants";
import osmtogeojson from "osmtogeojson";
import { useState } from "react";
import "mapbox-gl/dist/mapbox-gl.css";
import { readStreamableValue, useActions, useUIState } from "ai/rsc";
import { AI, ClientMessage } from "../actions";
import { nanoid } from "ai";

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

export default function OSM() {
  const [geojsonData, setGeojsonData] = useState(null);
  const [input, setInput] = useState("");
  const [conversation, setConversation] = useUIState<typeof AI>();
  const { sendMessage } = useActions<typeof AI>();

  const queryOsm = async (content: string) => {
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
  };

  const handleSubmit = async () => {
    setConversation((prev: ClientMessage[]) => [
      ...prev,
      { role: "user", display: input, id: nanoid() },
    ]);
    setInput("");
    const [message, statusStream] = await sendMessage(input);
    setConversation((prev: ClientMessage[]) => [...prev, message]);
    for await (const value of readStreamableValue(statusStream)) {
      if (value?.kind == "done") {
        await queryOsm(value.value);
      }
    }
  };

  const layers = [
    geojsonData &&
      new GeoJsonLayer({
        id: "geojson-layer",
        data: geojsonData,
        pickable: true,
        stroked: true,
        filled: true,
        extruded: true,
        pointType: "circle",
        getFillColor: [255, 165, 0, 200], // Changed to a more orangy color
        getLineColor: [255, 255, 255],
        getPointRadius: 10,
        pointRadiusMinPixels: 3,
        getLineWidth: 2,
        lineWidthMinPixels: 3,
        getElevation: 30,
      }),
  ];

  return (
    <div className="w-full h-screen flex flex-row gap-3">
      <div className="flex-1 flex flex-col overflow-hidden justify-start">
        <ChatMessages />
        <Chatbox
          handleSubmit={handleSubmit}
          handleInputChange={(e) => setInput(e.target.value)}
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
