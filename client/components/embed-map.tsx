"use client";
import React from "react";
import { GOOGLE_MAPS_API_KEY } from "@/lib/constants";
import { PurePoint } from "@/lib/geo";
import { ViewPanelType } from "@/app/sift/siftStore";

type EmbedMapProps = {
  coord: PurePoint;
  viewType: ViewPanelType;
  panoId?: string;
  autofocus?: boolean;
};

const DEFAULT_ZOOM = 20;

const getParams = ({ panoId, coord, viewType }: EmbedMapProps) => {
  if (viewType == "streetview") {
    return (
      `streetview?key=${GOOGLE_MAPS_API_KEY}&` +
      (panoId ? `pano=${panoId}` : `location=${coord?.lat},${coord?.lon}`)
    );
  }
  return `view?key=${GOOGLE_MAPS_API_KEY}&center=${coord?.lat},${
    coord?.lon
  }&zoom=${DEFAULT_ZOOM}&maptype=${
    viewType == "satellite" ? "satellite" : "roadmap"
  }`;
};

const EmbedMap: React.FC<EmbedMapProps> = (props) => {
  const src = `https://www.google.com/maps/embed/v1/${getParams(props)}`;
  return (
    <iframe
      src={src}
      className="w-full h-full relative"
      loading="lazy"
      autoFocus={props.autofocus}
    ></iframe>
  );
};

export default EmbedMap;
