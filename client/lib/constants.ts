import { MapViewState } from "deck.gl";

const API_URL = process.env.NEXT_PUBLIC_API_URL!;
const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN!;
const GOOGLE_MAPS_API_KEY = process.env.NEXT_PUBLIC_GMAPS_API!;

export const INITIAL_VIEW_STATE: MapViewState = {
  longitude: -122.41669,
  latitude: 37.7853,
  zoom: 1,
};

export { API_URL, MAPBOX_TOKEN, GOOGLE_MAPS_API_KEY };
