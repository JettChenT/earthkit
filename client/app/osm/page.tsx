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
import { AI, ClientMessage } from "./actions";
import { unstable_noStore as noStore } from "next/cache";
import { nanoid } from "ai";
import { bbox } from "@turf/bbox";
import { Orama } from "@orama/orama";
import { initializeDb, schema, searchDb } from "./searchSuggestions";

export const dynamic = "force-dynamic";
export const maxDuration = 30;

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

const getOsmPart = (content: string) => {
  return content.split("```overpassql\n").at(-1)?.split("\n```").at(0);
};

const queryOsm = async (osm_codeblock: string) => {
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
  noStore();
  const [geojsonData, setGeojsonData] = useState<GeoJSON.FeatureCollection>({
    type: "FeatureCollection",
    features: [],
  });
  const [input, setInput] = useState("");
  const [conversation, setConversation] = useUIState<typeof AI>();
  const { sendMessage } = useActions<typeof AI>();
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [db, setDb] = useState<Orama<typeof schema> | null>(null);

  const updateConversation = (id: string, newData: Partial<ClientMessage>) => {
    setConversation((prevConversation) =>
      prevConversation.map((msg) => {
        if (msg.id === id) {
          return { ...msg, ...newData };
        }
        return msg;
      })
    );
  };

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
        const osm_codeblock = getOsmPart(progress.value);
        if (!osm_codeblock) break;
        updateConversation(generation_id, {
          lowerIndicators: [
            <DummyProgressIndicator>
              Querying Overpass Turbo...
            </DummyProgressIndicator>,
          ],
        });
        const geojson = await queryOsm(osm_codeblock);
        if (geojson) {
          updateConversation(generation_id, {
            lowerIndicators: [<ResultsDisplay feats={geojson} />],
          });
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
          handleInputChange={(newInput) => {
            setInput(newInput);
          }}
          input={input}
          db={db}
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

function DummyProgressIndicator({ children }: { children: React.ReactNode }) {
  return (
    <div className="h-3 w-3 bg-gray-300 rounded-full animate-pulse">
      {children}
    </div>
  );
}

function ResultsDisplay({ feats }: { feats: GeoJSON.FeatureCollection }) {
  return (
    <div className="h-3 w-3 bg-gray-300 rounded-full animate-pulse">
      I gotcha some results
    </div>
  );
}
