"use client";
import EmbedMap from "@/components/embed-map";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ViewPanel from "./viewPanel";
import Table from "./table";
import { useComb } from "../../lib/combStore";
import LablView from "./lablView";
import { GeoImport } from "./geo-import";

function Panels() {
  return (
    <ResizablePanelGroup direction="horizontal">
      <ResizablePanel defaultSize={70}>
        <ResizablePanelGroup direction="vertical">
          <ResizablePanel defaultSize={60}>
            <ViewPanel />
          </ResizablePanel>
          <ResizableHandle />
          <ResizablePanel defaultSize={40}>
            <LablView />
          </ResizablePanel>
        </ResizablePanelGroup>
      </ResizablePanel>
      <ResizableHandle />
      <ResizablePanel defaultSize={30}>
        <Table />
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}

export default function Comb() {
  return <Panels />;
}
