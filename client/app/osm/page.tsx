"use client";

import { Chatbox } from "@/components/chatbox";
import { ChatMessages } from "@/components/chat-messages";
import ky from "ky";
import DeckGL, {
  FlyToInterpolator,
  GeoJsonLayer,
  WebMercatorViewport,
} from "deck.gl";
import { Map } from "react-map-gl";
import { INITIAL_VIEW_STATE, MAPBOX_TOKEN } from "@/lib/constants";
import osmtogeojson from "osmtogeojson";
import { useEffect, useState } from "react";
import "mapbox-gl/dist/mapbox-gl.css";
import { readStreamableValue, useActions, useUIState } from "ai/rsc";
import { AI, ClientMessage } from "../actions";
import { nanoid } from "ai";
import { bbox } from "@turf/bbox";
import { Orama } from "@orama/orama";
import { initializeDb, schema, searchDb } from "./searchSuggestions";

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

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
  return osmtogeojson(result);
};

export default function OSM() {
  const [geojsonData, setGeojsonData] = useState<GeoJSON.FeatureCollection>({
    type: "FeatureCollection",
    features: [],
  });
  const [input, setInput] = useState("");
  const [conversation, setConversation] = useUIState<typeof AI>();
  const { sendMessage } = useActions<typeof AI>();
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [db, setDb] = useState<Orama<typeof schema>>();

  useEffect(() => {
    initializeDb().then((db) => setDb(db));
  }, []);

  const handleSubmit = async () => {
    setConversation((prev: ClientMessage[]) => [
      ...prev,
      { role: "user", content: input, id: nanoid() },
    ]);
    setInput("");
    const { textStream, upperIndicator, progressStream } = await sendMessage(
      input
    );
    const generation_id = nanoid();
    setConversation((prev: ClientMessage[]) => [
      ...prev,
      {
        role: "assistant",
        content: "",
        upperIndicator: upperIndicator,
        id: generation_id,
      },
    ]);
    for await (const value of readStreamableValue(textStream)) {
      setConversation((prevConversation) =>
        prevConversation.map((msg) => {
          if (msg.id === generation_id) {
            return { ...msg, content: msg.content + value };
          }
          return msg;
        })
      );
    }
    for await (const progress of readStreamableValue(progressStream)) {
      console.log(progress);
      if (progress?.kind === "done") {
        const geojson = await queryOsm(progress.value);
        if (geojson) {
          setGeojsonData(geojson);
          const [minLng, minLat, maxLng, maxLat] = bbox(geojson);
          const vp = layer.context.viewport as WebMercatorViewport;
          const { longitude, latitude, zoom } = vp.fitBounds(
            [
              [minLng, minLat],
              [maxLng, maxLat],
            ],
            { padding: 100 }
          );
          setViewState({
            longitude,
            latitude,
            zoom: Math.max(zoom - 1, 2),
            transitionInterpolator: new FlyToInterpolator({ speed: 2 }),
            transitionDuration: "auto",
          });
        }
      }
    }
  };

  const layer = new GeoJsonLayer({
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
  });

  return (
    <div className="w-full h-screen flex flex-row gap-3">
      <div className="flex-1 flex flex-col overflow-hidden justify-start">
        <ChatMessages />
        <Chatbox
          handleSubmit={handleSubmit}
          handleInputChange={(e) => {
            setInput(e.target.value);
            if (db) {
              searchDb(db, e.target.value).then((suggestions) => {
                console.log(suggestions);
              });
            }
          }}
          input={input}
        />
      </div>
      <div className="flex-1 p-3">
        <div className="h-full relative">
          <DeckGL initialViewState={viewState} controller layers={[layer]}>
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
