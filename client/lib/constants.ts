import { MapViewState } from "deck.gl";

const API_URL = process.env.NEXT_PUBLIC_API_URL!;
const AI_API_URL = process.env.NEXT_PUBLIC_AI_SERV_ADDR!;
const RAW_API_URL = process.env.NEXT_PUBLIC_RAW_API_URL!;
const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN!;
const GOOGLE_MAPS_API_KEY = process.env.NEXT_PUBLIC_GMAPS_API!;
const IS_LOCAL = process.env.IS_LOCAL === "true";

export const INITIAL_VIEW_STATE: MapViewState = {
  longitude: -122.41669,
  latitude: 37.7853,
  zoom: 1,
};

export const DEFAULT_MAP_STYLE =
  "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

export {
  API_URL,
  AI_API_URL,
  RAW_API_URL,
  MAPBOX_TOKEN,
  GOOGLE_MAPS_API_KEY,
  IS_LOCAL,
};
