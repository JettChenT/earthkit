"use client";
import React from "react";
import { GOOGLE_MAPS_API_KEY } from "@/lib/constants";
import { Point } from "@/lib/geo";

type StreetViewPanoProps = {
  panoId?: string;
  coord?: Point;
};

const StreetViewPano: React.FC<StreetViewPanoProps> = ({ panoId, coord }) => {
  const locChnk = panoId
    ? `pano=${panoId}`
    : `location=${coord?.lat},${coord?.lon}`;
  const src = `https://www.google.com/maps/embed/v1/streetview?key=${GOOGLE_MAPS_API_KEY}&${locChnk}`;
  return (
    <iframe
      src={src}
      className="w-full h-full relative"
      loading="lazy"
    ></iframe>
  );
};

export default StreetViewPano;
