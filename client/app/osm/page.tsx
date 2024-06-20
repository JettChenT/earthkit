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
import { Button } from "@/components/ui/button";
import { importData, parseGeoJsonImport } from "../sift/inout";
import { downloadContent } from "@/lib/utils";
import { center } from "@turf/center";
import { useSift } from "@/app/sift/siftStore";
import { useRouter } from "next/navigation";
import { overpassJson } from "@/lib/overpass";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

export const dynamic = "force-dynamic";
export const maxDuration = 30;

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

export const getOsmPart = (content: string) => {
  const match = content.match(/```overpassql\n([\s\S]*?)\n```/);
  return match ? match[1] : null;
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

  const handleSubmit = async (
    user_input: string,
    sys_results: string[] = []
  ) => {
    setConversation((prev: ClientMessage[]) => [
      ...prev,
      { role: "user", content: user_input, id: nanoid() },
    ]);
    const { textStream, upperIndicator, progressStream } = await sendMessage(
      user_input,
      sys_results
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
            <DummyProgressIndicator key={nanoid()}>
              Querying Overpass Turbo...
            </DummyProgressIndicator>,
          ],
        });
        const geojson = await overpassJson(osm_codeblock)
          .then((res) => {
            console.log("results", res);
            return osmtogeojson(res);
          })
          .catch((e) => {
            updateConversation(generation_id, {
              lowerIndicators: [
                <ErrorDisplay
                  key={nanoid()}
                  errorHeader={"Overpass Turbo Query Error"}
                  errorDetail={e.message}
                  onFix={() => {
                    handleSubmit("Please fix this error", [
                      `Error: ${e.message}`,
                    ]);
                  }}
                />,
              ],
            });
            return null;
          });
        console.log("parsed geojson", geojson);
        if (geojson) {
          updateConversation(generation_id, {
            lowerIndicators: [
              <ResultsDisplay feats={geojson} key={nanoid()} />,
            ],
          });
          setGeojsonData(geojson);
          if (geojson.features.length == 0) return;
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
          handleSubmit={() => {
            handleSubmit(input);
            setInput("");
          }}
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
    <div className="bg-secondary rounded-md p-2 flex flex-row justify-between">
      {children}
    </div>
  );
}

function ErrorDisplay({
  errorHeader,
  errorDetail,
  onFix,
}: {
  errorHeader: string;
  errorDetail: string;
  onFix: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="w-full">
      <div className="p-2 bg-red-100 rounded-md flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-red-700">
            {errorHeader}
          </span>
          <div className="flex gap-1">
            <Button size="sm" variant="outline" onClick={onFix}>
              Fix
            </Button>
            <CollapsibleTrigger asChild>
              <Button size="sm" variant="outline">
                {isOpen ? "Hide Details" : "Show Details"}
              </Button>
            </CollapsibleTrigger>
          </div>
        </div>
        <CollapsibleContent>
          <textarea
            className="w-full mt-2 text-sm text-red-700 bg-black p-2 rounded-md h-64"
            readOnly
            value={errorDetail}
          />
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

function ResultsDisplay({ feats }: { feats: GeoJSON.FeatureCollection }) {
  const { addItems } = useSift();
  const router = useRouter();
  const featuresCount = feats.features.length;

  return (
    <div className="p-2 bg-secondary rounded-md flex items-center justify-between">
      <span className="text-sm font-medium">
        {featuresCount > 0
          ? `Fetched ${featuresCount} features`
          : "No features found"}
      </span>
      {featuresCount > 0 && (
        <div className="flex gap-1">
          <Button
            size={"sm"}
            onClick={() => {
              const res = parseGeoJsonImport(feats);
              console.log(res.items);
              addItems(res.items);
              router.push("/sift");
            }}
          >
            Sift
          </Button>
          <Button
            size={"sm"}
            onClick={() => {
              downloadContent(JSON.stringify(feats), "geojson");
            }}
          >
            Export
          </Button>
        </div>
      )}
    </div>
  );
}
