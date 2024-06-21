"use client";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { CornerDownLeft, Key, Pin, Tag } from "lucide-react";
import { MentionsInput, Mention, OnChangeHandlerFunc } from "react-mentions";
import { OSMOrama, searchDb, Document } from "@/app/osm/searchSuggestions";
import CodeMirror, {
  EditorState,
  ReactCodeMirrorRef,
  Text,
} from "@uiw/react-codemirror";
import {
  CompletionContext,
  autocompletion,
  CompletionResult,
  startCompletion,
} from "@codemirror/autocomplete";
import {
  MatchDecorator,
  WidgetType,
  Decoration,
  DecorationSet,
  EditorView,
  ViewPlugin,
  keymap,
  ViewUpdate,
} from "@codemirror/view";
import "codemirror/lib/codemirror.css";
import "codemirror/theme/material.css";
import React, { createRef, useRef } from "react";
import { createRoot } from "react-dom/client";
import { geoSearch } from "@/lib/nominatim";

function renderReactNode(
  node: React.ReactNode,
  parent: string = "div",
  className?: string
): HTMLElement {
  const div = document.createElement(parent);
  if (className) {
    div.className = className;
  }
  const root = createRoot(div);
  root.render(node);
  return div;
}

function OSMSuggestionPill({
  type,
  name,
}: {
  type: "key" | "tag" | "location";
  name: string;
}) {
  const tpName = type == "key" ? "key" : "feat";
  return (
    <div
      className={`inline-flex items-baseline px-1 mt-1 rounded-sm shadow-sm  ${
        type == "key"
          ? "bg-blue-100 text-blue-800"
          : type == "location"
          ? "bg-red-100 text-red-800"
          : "bg-green-100 text-green-800"
      }`}
    >
      <span className="font-bold mr-1">
        {type == "key" ? (
          <Key className="size-3 inline-block" />
        ) : type == "location" ? (
          <Pin className="size-3 inline-block" />
        ) : (
          <Tag strokeWidth={3} className="size-3 inline-block" />
        )}
      </span>
      <span>{name}</span>
    </div>
  );
}

class OSMSuggestionWidget extends WidgetType {
  type: string;
  name: string;
  constructor(type: string, name: string) {
    super();
    this.type = type;
    this.name = name;
  }
  toDOM(view: EditorView): HTMLElement {
    return renderReactNode(
      <OSMSuggestionPill type={this.type as any} name={this.name} />,
      "span",
      "font-bold"
    );
  }
}

class LocationSuggestionWidget extends WidgetType {
  name: string;
  constructor(name: string) {
    super();
    this.name = name;
  }
  toDOM(view: EditorView): HTMLElement {
    return renderReactNode(
      <OSMSuggestionPill type="location" name={this.name} />,
      "span",
      "font-bold"
    );
  }
}

const osmSuggestionMatcher = new MatchDecorator({
  regexp: /\(OSM (\w+): `([^\`]+)`\)/g,
  decoration: (match) =>
    Decoration.replace({
      widget: new OSMSuggestionWidget(match[1], match[2]),
    }),
});

const locationSuggestionMatcher = new MatchDecorator({
  regexp: /\(Entity osm_id=(\d+);area_id=(\d+);(\w+): `([^\`]+)`\)/g,
  decoration: (match) =>
    Decoration.replace({
      widget: new LocationSuggestionWidget(match[4]),
    }),
});

export const osm_placeholders = ViewPlugin.fromClass(
  class {
    placeholders: DecorationSet;
    constructor(view: EditorView) {
      this.placeholders = osmSuggestionMatcher.createDeco(view);
    }
    update(update: ViewUpdate) {
      this.placeholders = osmSuggestionMatcher.updateDeco(
        update,
        this.placeholders
      );
    }
  },
  {
    decorations: (instance) => instance.placeholders,
    provide: (plugin) =>
      EditorView.atomicRanges.of((view) => {
        return view.plugin(plugin)?.placeholders || Decoration.none;
      }),
  }
);

export const location_placeholders = ViewPlugin.fromClass(
  class {
    placeholders: DecorationSet;
    constructor(view: EditorView) {
      this.placeholders = locationSuggestionMatcher.createDeco(view);
    }
    update(update: ViewUpdate) {
      this.placeholders = locationSuggestionMatcher.updateDeco(
        update,
        this.placeholders
      );
    }
  },
  {
    decorations: (instance) => instance.placeholders,
    provide: (plugin) =>
      EditorView.atomicRanges.of((view) => {
        return view.plugin(plugin)?.placeholders || Decoration.none;
      }),
  }
);

interface ChatboxProps {
  handleSubmit: () => void;
  handleInputChange: (newInput: string) => void;
  input: string;
  db: OSMOrama | null;
}

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

export function Chatbox({
  handleSubmit,
  handleInputChange,
  input,
  db,
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

  return (
    <form
      onSubmit={handleSubmit}
      className="flex-none p-2 bg-white border rounded-md w-full mb-3"
    >
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
        <div className="flex ml-2">
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
        </div>
        <Button type="submit" variant="secondary" className="py-0" size={"sm"}>
          <CornerDownLeft className="size-3 font-bold h-3 w-3" />
        </Button>
      </div>
    </form>
  );
}
