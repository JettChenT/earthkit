import { MAPBOX_TOKEN } from "@/lib/constants";
import { SearchBox } from "@mapbox/search-js-react";
import {
  DeckGLRef,
  FlyToInterpolator,
  MapViewState,
  Viewport,
  WebMercatorViewport,
} from "deck.gl";
import { RefObject, useState } from "react";

interface ESearchBoxProps {
  setViewState: (viewport: MapViewState) => void;
  dglref: RefObject<DeckGLRef>;
}

const searchTypes = [
  "country",
  "region",
  "district",
  "postcode",
  "place",
  "locality",
  "neighborhood",
];
const searchTypesSet = new Set(searchTypes);

export function ESearchBox({ setViewState, dglref }: ESearchBoxProps) {
  const [value, setValue] = useState("");
  return (
    <form className="absolute left-3 top-3">
      {/* @ts-ignore */}
      <SearchBox
        placeholder="Go to Location"
        accessToken={MAPBOX_TOKEN}
        value={value}
        onChange={setValue}
        options={{
          poi_category: "place",
          poi_category_exclusions: "brand",
          // @ts-ignore
          types: searchTypesSet,
        }}
        onRetrieve={(e) => {
          let vp =
            dglref.current?.deck?.getViewports()[0] as WebMercatorViewport;
          let bbox = e.features[0].properties.bbox;
          if (!bbox) return;
          const { longitude, latitude, zoom } = vp.fitBounds(
            [
              [bbox[0], bbox[1]],
              [bbox[2], bbox[3]],
            ],
            { padding: 100 }
          );
          setViewState({
            longitude,
            latitude,
            zoom,
            transitionInterpolator: new FlyToInterpolator({ speed: 4 }),
            transitionDuration: "auto",
          });
        }}
      />
    </form>
  );
}
