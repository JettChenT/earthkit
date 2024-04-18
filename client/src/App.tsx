import { Map, useMap, useMapsLibrary } from "@vis.gl/react-google-maps";
import { useEffect } from "react";

export default function App() {
  const map = useMap();
  const svLibrary = useMapsLibrary("streetView");

  useEffect(() => {
    if (!svLibrary || !map) return;
    const cov = new svLibrary.StreetViewCoverageLayer();
    cov.setMap(map);
  }, [svLibrary, map]);

  return (
    <Map
      style={{ width: "100vw", height: "100vh" }}
      defaultCenter={{ lat: 22.54992, lng: 0 }}
      defaultZoom={3}
      gestureHandling={"greedy"}
      disableDefaultUI={true}
    />
  );
}
