"use client";

import { Chatbox } from "@/components/chatbox";
import { useChat } from "ai/react";

import { ChatMessages } from "@/components/chat-messages";
import ky from "ky";
import DeckGL from "deck.gl";
import { Map } from "react-map-gl";
import { INITIAL_VIEW_STATE, MAPBOX_TOKEN } from "@/lib/constants";
import osmtogeojson from "osmtogeojson";

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

export default function OSM() {
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
    },
  });
  return (
    <div className="w-full h-screen flex flex-row gap-3">
      <div className="w-1/2 h-screen flex flex-col">
        <ChatMessages messages={messages} />
        <Chatbox
          handleSubmit={handleSubmit}
          handleInputChange={handleInputChange}
          input={input}
        />
      </div>
      <div className="w-1/2 h-screen relative p-3">
        <div className="w-full h-full relative">
          <DeckGL initialViewState={INITIAL_VIEW_STATE} controller>
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
