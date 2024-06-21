"use client";
import { renderReactNode } from "@/lib/react_utils";
import {
  Decoration,
  DecorationSet,
  EditorView,
  MatchDecorator,
  ViewPlugin,
  ViewUpdate,
  WidgetType,
} from "@codemirror/view";
import "codemirror/lib/codemirror.css";
import "codemirror/theme/material.css";
import { Key, Pin, Tag } from "lucide-react";

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
