"use client";
import { OSMOrama, searchDb } from "@/app/osm/searchSuggestions";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CancellableImage } from "@/components/widgets/imageUpload";
import { geoSearch } from "@/lib/nominatim";
import { renderReactNode } from "@/lib/react_utils";
import { useUser } from "@clerk/nextjs";
import {
  CompletionContext,
  CompletionResult,
  autocompletion,
  startCompletion,
} from "@codemirror/autocomplete";
import { EditorView } from "@codemirror/view";
import CodeMirror, { ReactCodeMirrorRef } from "@uiw/react-codemirror";
import "codemirror/lib/codemirror.css";
import "codemirror/theme/material.css";
import { CornerDownLeft } from "lucide-react";
import React, { useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { Model } from "./actions";
import { location_placeholders, osm_placeholders } from "./cm_common";
import { useOsmGlobs } from "./osmState";
import { ImageUploadButton } from "@/components/widgets/imageUpload";
import { Image as ImageIcon } from "lucide-react";

const locationCompletion = async (
  context: CompletionContext
): Promise<CompletionResult | null> => {
  let word = context.matchBefore(/\@[\w ]*/);
  if (!word) return null;
  if (word.text.length == 1) {
    return {
      from: word.from,
      filter: false,
      options: [
        {
          label: "Enter query to search for location...",
        },
      ],
    };
  }
  if (word.from == word.to && !context.explicit) return null;
  const suggestions = await geoSearch(word.text.slice(1));
  return {
    from: word.from,
    filter: false,
    options: suggestions.map((suggestion) => {
      return {
        label: `(Entity osm_id=${suggestion.osm_id};area_id=${suggestion.area_id};${suggestion.class}: \`${suggestion.name}\`)`,
        displayLabel: `${suggestion.display_name}`,
        type: suggestion.class == "administrative" ? "keyword" : "variable",
        info: (cmpl) => {
          return renderReactNode(
            <div className="text-sm">
              <div className="font-bold">{suggestion.name}</div>
              <div className="text-xs text-gray-500">
                {suggestion.display_name}
              </div>
              <div className="text-xs text-gray-500">{suggestion.type}</div>
              <div className="text-xs text-gray-500">{suggestion.osm_type}</div>
              <div className="text-xs text-gray-500">
                {suggestion.address_type}
              </div>
            </div>
          );
        },
      };
    }),
  };
};

interface ChatboxProps {
  handleSubmit: () => void;
  handleInputChange: (newInput: string) => void;
  input: string;
  db: OSMOrama | null;
  images: string[];
  setImages: (images: string[]) => void;
}

export function Chatbox({
  handleSubmit,
  handleInputChange,
  input,
  db,
  images,
  setImages,
}: ChatboxProps) {
  const handleKeyDown = (
    event:
      | React.KeyboardEvent<HTMLTextAreaElement>
      | React.KeyboardEvent<HTMLInputElement>
      | React.KeyboardEvent<HTMLDivElement>
  ) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  };

  const osmCompletion = async (
    context: CompletionContext
  ): Promise<CompletionResult | null> => {
    let word = context.matchBefore(/\#\w*/);
    if (!word) return null;
    if (word.from == word.to && !context.explicit) return null;
    if (!db) return null;
    const suggestions = await searchDb(db, word.text.slice(1));
    return {
      from: word.from,
      filter: false,
      options: suggestions.hits.map((suggestion) => {
        return {
          label: `(OSM ${suggestion.document.type}: \`${suggestion.document.name}\`)`,
          displayLabel: `${suggestion.document.name}`,
          type: suggestion.document.type == "key" ? "keyword" : "variable",
          info: (cmpl) => {
            const description = document.createElement("div");
            const rt = createRoot(description);
            rt.render(
              <div className="text-sm">
                <div className="font-bold">{suggestion.document.name}</div>
                <div className="text-xs text-gray-500">
                  OSM{" "}
                  <a
                    href={
                      suggestion.document.type == "key"
                        ? "https://taginfo.openstreetmap.org/keys"
                        : "https://taginfo.openstreetmap.org/features"
                    }
                    className="text-blue-600 underline"
                  >
                    {suggestion.document.type == "key" ? "Key" : "Feature"}
                  </a>
                </div>
                <div className="text-xs text-gray-500">
                  {suggestion.document.description}
                </div>
              </div>
            );
            return description;
          },
        };
      }),
    };
  };

  const completionExt = autocompletion({
    aboveCursor: true,
    override: [osmCompletion, locationCompletion],
  });

  let cmref = useRef<ReactCodeMirrorRef | null>(null);

  const handleShortcut = (shortcut: string) => {
    cmref.current?.view?.dispatch({
      changes: {
        from: cmref.current.view.state.selection.main.head,
        insert: shortcut,
      },
      selection: {
        anchor: cmref.current.view.state.selection.main.head + 1,
        head: cmref.current.view.state.selection.main.head + 1,
      },
    });
    cmref.current?.view?.focus();
    if (cmref.current?.view) {
      startCompletion(cmref.current.view);
    }
  };

  let { setModel } = useOsmGlobs();

  let { isSignedIn } = useUser();
  useEffect(() => {
    setModel("gpt-4o");
  }, [setModel]);

  return (
    <div className="flex-none p-2 bg-white border rounded-md w-full mb-3">
      <div className="flex flex-row justify-start gap-1 pl-3">
        {images.map((image, idx) => (
          <CancellableImage
            key={idx}
            image={image}
            className="mb-2"
            onCancel={() => {
              setImages(images.filter((_, i) => i !== idx));
            }}
          />
        ))}
      </div>
      <div className="flex-1 pl-2">
        <CodeMirror
          value={input}
          ref={cmref}
          onChange={(value) => {
            if (value.includes("\n")) {
              handleSubmit();
              return;
            }
            handleInputChange(value);
          }}
          height="50px"
          autoFocus={true}
          placeholder="Describe a query... Use `#` for OSM tags/features, `@` for locations/areas."
          extensions={[
            EditorView.theme({
              "&.cm-focused": {
                outline: "none",
              },
            }),
            EditorView.lineWrapping,
            completionExt,
            osm_placeholders,
            location_placeholders,
          ]}
          basicSetup={{
            lineNumbers: false,
            foldGutter: false,
            highlightActiveLine: false,
          }}
        />
      </div>
      <div className="flex items-center justify-between">
        <div className="flex ml-2 items-center">
          <Button
            variant="ghost"
            size="sm"
            className="text-xs text-secondary-foreground px-1"
            type="button"
            onClick={() => {
              handleShortcut("#");
            }}
          >
            # Features
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-xs text-secondary-foreground px-1"
            type="button"
            onClick={() => {
              handleShortcut("@");
            }}
          >
            @ Locations
          </Button>
          <ImageUploadButton
            onSetImage={(image) => {
              setImages([...images, image]);
            }}
          >
            <Button
              variant="ghost"
              size="sm"
              className="text-xs text-secondary-foreground px-1"
              type="button"
            >
              <ImageIcon className="size-3 inline-block mr-1" /> Image
            </Button>
          </ImageUploadButton>
          {!isSignedIn && (
            <span className="text-xs text-gray-500 ml-2 inline-block">
              Not signed in; limited to 3 queries per hour.
            </span>
          )}
        </div>
        <Button
          type="button"
          variant="secondary"
          className="py-0"
          size={"sm"}
          onClick={() => {
            if (input.length > 0) {
              handleSubmit();
            }
          }}
        >
          <CornerDownLeft className="size-3 font-bold h-3 w-3" />
        </Button>
      </div>
    </div>
  );
}
