"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { ViewPanelType, useComb } from "../../lib/combStore";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import EmbedMap from "@/components/embed-map";

export default function ViewPanel() {
  const { viewPanelState, setViewPanelState } = useComb();
  const currentItem = useComb((state) => state.getSelected());
  if (!currentItem) return null;
  return (
    <Tabs
      value={viewPanelState}
      onValueChange={(value) => setViewPanelState(value as ViewPanelType)}
      className="h-full flex flex-col gap-1"
    >
      <div>
        <TabsList>
          <TabsTrigger value="map">Map</TabsTrigger>
          <TabsTrigger value="streetview">Street View</TabsTrigger>
          <TabsTrigger value="satellite">Satellite</TabsTrigger>
        </TabsList>
      </div>
      <div className="grow">
        <EmbedMap
          panoId={currentItem.aux?.panoId}
          coord={currentItem.coord}
          viewType={viewPanelState}
        />
      </div>
    </Tabs>
  );
}
