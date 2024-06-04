"use client";
import StreetViewPano from "@/components/streetview-pano";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ViewPanel from "./viewPanel";
import Table from "./table";
import { useComb } from "./combStore";
import { useHotkeys } from "react-hotkeys-hook";
import { Key } from "ts-key-enum";

export default function Comb() {
  const { idxDelta } = useComb();
  useHotkeys(["j", Key.ArrowDown], () => {
    idxDelta(1);
  });

  useHotkeys(["k", Key.ArrowUp], () => {
    idxDelta(-1);
  });

  return (
    <ResizablePanelGroup direction="horizontal">
      <ResizablePanel defaultSize={60}>
        <ResizablePanelGroup direction="vertical">
          <ResizablePanel defaultSize={60}>
            <ViewPanel />
          </ResizablePanel>
          <ResizableHandle />
          <ResizablePanel defaultSize={40}>
            This is the labeling view
          </ResizablePanel>
        </ResizablePanelGroup>
      </ResizablePanel>
      <ResizableHandle />
      <ResizablePanel defaultSize={50}>
        <Table />
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}