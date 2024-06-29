"use client";

import { ChatMessages } from "@/app/osm/chat-messages";
import { Chatbox } from "@/app/osm/chatbox";
import { useSift } from "@/app/sift/siftStore";
import { CommandBar, useListeners } from "@/components/kbar";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  DEFAULT_MAP_STYLE,
  INITIAL_VIEW_STATE,
  MAPBOX_TOKEN,
} from "@/lib/constants";
import { overpassJson } from "@/lib/overpass";
import { downloadContent } from "@/lib/utils";
import { Orama } from "@orama/orama";
import { bbox } from "@turf/bbox";
import { ImagePart, nanoid } from "ai";
import { readStreamableValue, useActions, useUIState } from "ai/rsc";
import DeckGL, {
  FlyToInterpolator,
  GeoJsonLayer,
  WebMercatorViewport,
} from "deck.gl";
import { ArrowRight, Download, Trash2 } from "lucide-react";
import "mapbox-gl/dist/mapbox-gl.css";
import { unstable_noStore as noStore } from "next/cache";
import { useRouter } from "next/navigation";
import osmtogeojson from "osmtogeojson";
import { useEffect, useState } from "react";
import { Map } from "react-map-gl";
import { parseGeoJsonImport } from "../sift/inout";
import { AI, ClientMessage, Model } from "./actions";
import { useOsmGlobs } from "./osmState";
import { initializeDb, schema } from "./searchSuggestions";
import { useSWRConfig } from "swr";
import { useAuth } from "@clerk/nextjs";

export const dynamic = "force-dynamic";
export const maxDuration = 30;

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
  const [images, setImages] = useState<string[]>([]);
  const [conversation, setConversation] = useUIState<typeof AI>();
  const { sendMessage } = useActions<typeof AI>();
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [db, setDb] = useState<Orama<typeof schema> | null>(null);
  const { setLatestGeneration } = useOsmGlobs();
  const { mutate } = useSWRConfig();
  const { isLoaded, isSignedIn } = useAuth();

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
    sys_results: string[] = [],
    images: string[] = []
  ) => {
    const image_content: ImagePart[] = images.map((image) => ({
      type: "image",
      image: image,
    }));
    setConversation((prev: ClientMessage[]) => [
      ...prev,
      {
        role: "user",
        content: [...image_content, { type: "text", text: user_input }],
        id: nanoid(),
      },
    ]);
    const { textStream, upperIndicator, progressStream } = await sendMessage(
      user_input,
      sys_results,
      images,
      "gpt-4o"
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
          if (msg.id === generation_id && value) {
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
          setLatestGeneration(generation_id);
          updateConversation(generation_id, {
            lowerIndicators: [
              <ResultsDisplay
                feats={geojson}
                generation_id={generation_id}
                key={nanoid()}
              />,
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
          if (isLoaded && isSignedIn) mutate("/api/usage");
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
            handleSubmit(input, [], images);
            setInput("");
            setImages([]);
          }}
          handleInputChange={(newInput) => {
            setInput(newInput);
          }}
          images={images}
          setImages={setImages}
          input={input}
          db={db}
        />
      </div>
      <div className="flex-1 p-3">
        <div className="h-full relative">
          <DeckGL initialViewState={viewState} controller layers={[layer]}>
            <Map
              mapStyle={DEFAULT_MAP_STYLE}
              mapboxAccessToken={MAPBOX_TOKEN}
            ></Map>
          </DeckGL>
        </div>
      </div>
      <CommandBar
        commands={[
          {
            type: "CommandGroupData",
            children: [
              {
                type: "CommandItemData",
                action: () => {
                  setConversation([]);
                  setGeojsonData({
                    type: "FeatureCollection",
                    features: [],
                  });
                },
                display: "Clear Chat",
                icon: <Trash2 className="size-5" />,
              },
              {
                type: "CommandItemData",
                event: "SiftLatest",
                display: "Sift Through the Latest Results",
                disabled: geojsonData.features.length == 0,
                icon: <ArrowRight className="size-5" />,
              },
              {
                type: "CommandItemData",
                event: "ExportLatest",
                display: "Export Latest Results to GeoJSON",
                disabled: geojsonData.features.length == 0,
                icon: <Download className="size-5" />,
              },
            ],
            heading: "Overpass Query Actions",
          },
        ]}
      />
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

function ResultsDisplay({
  feats,
  generation_id,
}: {
  feats: GeoJSON.FeatureCollection;
  generation_id: string;
}) {
  const { addItems } = useSift();
  const router = useRouter();
  const featuresCount = feats.features.length;
  const toSift = () => {
    const res = parseGeoJsonImport(feats);
    console.log(res.items);
    addItems(res.items);
    router.push("/sift");
  };

  const toExport = () => {
    downloadContent(JSON.stringify(feats), "geojson");
  };

  let { latestGeneration } = useOsmGlobs();

  useListeners([
    {
      event: "SiftLatest",
      handler: () => {
        if (latestGeneration == generation_id) toSift();
      },
    },
    {
      event: "ExportLatest",
      handler: () => {
        if (latestGeneration == generation_id) toExport();
      },
    },
  ]);

  return (
    <div className="p-2 bg-secondary rounded-md flex items-center justify-between">
      <span className="text-sm font-medium">
        {featuresCount > 0
          ? `Fetched ${featuresCount} features`
          : "No features found"}
      </span>
      {featuresCount > 0 && (
        <div className="flex gap-1">
          <Button size={"sm"} onClick={toSift}>
            Sift
          </Button>
          <Button size={"sm"} onClick={toExport}>
            Export
          </Button>
        </div>
      )}
    </div>
  );
}
