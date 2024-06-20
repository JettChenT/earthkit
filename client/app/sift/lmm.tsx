"use client";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { forwardRef, useState } from "react";
import { cn } from "@/lib/utils";
import { useSift } from "./siftStore";
import ky from "ky";
import { API_URL } from "@/lib/constants";
import { toast } from "sonner";
import { ingestStream } from "@/lib/rpc";

export type Dependency = {
  satellite: boolean;
  streetview: boolean;
  basicmap: boolean;
  target_image: boolean;
};

export type OutputFormat = "text" | "number" | "boolean";

export const CustomExtraction = forwardRef<HTMLDivElement>((props, ref) => {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [prompt, setPrompt] = useState("");
  const [dependencies, setDependencies] = useState<Dependency>({
    satellite: false,
    streetview: false,
    basicmap: false,
    target_image: false,
  });
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("text");
  let { target_image, getCoords, setCols, updateItemResults } = useSift();

  const toggleDependency = (key: keyof Dependency) => {
    setDependencies((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const triggerAction = async () => {
    const payload = {
      dependencies,
      prompt: `${title}: ${prompt == "" ? `Extract ${prompt}` : prompt}`,
      output_type: outputFormat,
      coords: getCoords(),
      target_image,
      config: {
        model: "gpt-4o",
      },
    };

    setCols((cols) => [
      ...cols,
      {
        type: outputFormat == "number" ? "NumericalCol" : "TextCol",
        accessor: `${title}.answer`,
        header: title,
        isFunCall: true,
      },
    ]);

    const res = await ky.post(`${API_URL}/lmm/streaming`, {
      timeout: false,
      json: payload,
    });

    if (!res.ok || !res.body) {
      console.error(res);
      toast.error("Failed to get results");
      return;
    }

    for await (const chunk of ingestStream(res)) {
      if (chunk.type == "ResultsUpdate") {
        updateItemResults(chunk, title);
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger>
        <div ref={ref} className="hidden" />
      </DialogTrigger>
      <DialogContent>
        <Label>Title</Label>
        <Input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Enter the title for the extraction"
        />
        <Label>Model Inputs</Label>
        <div className="flex flex-row gap-2 w-full m-2">
          <SelectionBox
            selected={dependencies.satellite}
            onSelect={() => toggleDependency("satellite")}
            bgImage="https://maps.gstatic.com/tactile/layerswitcher/ic_satellite-2x.png"
          >
            Satellite
          </SelectionBox>
          <SelectionBox
            selected={dependencies.streetview}
            bgImage="https://maps.gstatic.com/tactile/layerswitcher/ic_streetview_colors2-2x.png"
            onSelect={() => toggleDependency("streetview")}
          >
            Streetview
          </SelectionBox>
          <SelectionBox
            selected={dependencies.basicmap}
            onSelect={() => toggleDependency("basicmap")}
            bgImage="https://maps.gstatic.com/tactile/layerswitcher/ic_default_colors2-2x.png"
          >
            Administrative
          </SelectionBox>
          <SelectionBox
            selected={dependencies.target_image}
            onSelect={() => toggleDependency("target_image")}
            bgImage={target_image || ""}
          >
            Target Image
          </SelectionBox>
        </div>
        <Label>Output format</Label>
        <div className="flex flex-row gap-2 w-full m-2">
          <SelectionBox
            selected={outputFormat === "text"}
            onSelect={() => setOutputFormat("text")}
            className="h-12"
          >
            Text
          </SelectionBox>
          <SelectionBox
            selected={outputFormat === "number"}
            onSelect={() => setOutputFormat("number")}
            className="h-12"
          >
            Number
          </SelectionBox>
          <SelectionBox
            selected={outputFormat === "boolean"}
            onSelect={() => setOutputFormat("boolean")}
            className="h-12"
          >
            Boolean
          </SelectionBox>
        </div>
        <Label>Description</Label>
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the data you want to extract"
        />
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button variant="default" onClick={triggerAction}>
            Add Column
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
});

function SelectionBox({
  selected,
  onSelect,
  className,
  bgImage,
  children,
}: {
  selected: boolean;
  onSelect: () => void;
  className?: string;
  bgImage?: string;
  children?: React.ReactNode;
}) {
  return (
    <div
      onClick={onSelect}
      className={cn(
        `h-24 w-full border rounded-md p-2 cursor-pointer transition-all duration-200 flex items-start justify-start ${
          selected
            ? "bg-blue-400 border-blue-700 ring-2"
            : "bg-white border-gray-300"
        }`,
        className
      )}
      style={{ backgroundImage: `url(${bgImage})`, backgroundSize: "cover" }}
    >
      <div className="flex items-center gap-2">
        <span>{children}</span>
      </div>
    </div>
  );
}
