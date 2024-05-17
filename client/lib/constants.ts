import { MapViewState } from "deck.gl";

const API_URL = process.env.NEXT_PUBLIC_API_URL!;
const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN!;

export const INITIAL_VIEW_STATE: MapViewState = {
  longitude: -122.41669,
  latitude: 37.7853,
  zoom: 1,
};

export { API_URL, MAPBOX_TOKEN };
