"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { ViewPanelType, useComb } from "./combStore";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import StreetViewPano from "@/components/streetview-pano";

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
        <TabsContent value="map" className="h-full" asChild>
          <Skeleton />
        </TabsContent>
        <TabsContent value="streetview" className="h-full" asChild>
          <StreetViewPano
            panoId={currentItem.panoId}
            coord={currentItem.coord}
          />
        </TabsContent>
        <TabsContent value="satellite" className="h-full" asChild>
          <Skeleton />
        </TabsContent>
      </div>
    </Tabs>
  );
}
