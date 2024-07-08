import { Button } from "@/components/ui/button";
import { LabelType, useSift } from "./siftStore";
import { useHotkeys, Keys } from "react-hotkeys-hook";
import { useRef, useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import React from "react";
import ImageUpload from "@/components/widgets/imageUpload";
import { cn } from "@/lib/utils";
import { getPillColorCn, statusToPillColor } from "@/components/pill";
import { Label } from "@/components/ui/label";
import Kbd from "@/components/keyboard";

const LabelButton = ({
  status,
  hotkey,
  explainer,
  disabled,
}: {
  status: LabelType;
  hotkey?: string[];
  explainer?: string;
  disabled?: boolean;
}) => {
  const { setIdxData } = useSift();
  const btnRef = useRef<HTMLButtonElement>(null);
  const [hl, setHl] = useState(false);
  useHotkeys(hotkey ?? "", () => {
    btnRef.current?.click();
  });

  const triggerhl = () => {
    setHl(true);
    setTimeout(() => {
      setHl(false);
    }, 150);
  };

  const selCn = getPillColorCn(statusToPillColor(status));

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            className={cn(
              "py-8 w-32 transition-all duration-100",
              hl && `hover:${selCn} ${selCn}`
            )}
            onClick={() => {
              triggerhl();
              setIdxData({ status });
            }}
            ref={btnRef}
            disabled={disabled}
          >
            {status}
          </Button>
        </TooltipTrigger>
        <TooltipContent side="top">
          Shortcuts:{" "}
          {hotkey?.map((key, index) => (
            <span key={index}>
              <Kbd key={index}>{key}</Kbd>
              {index < hotkey.length - 1 && ", "}
            </span>
          ))}
          {explainer && <div>{explainer}</div>}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default function LablView() {
  const { setIdxData, setTargetImage, target_image } = useSift();
  const [imageExpanded, setImageExpanded] = useState(false);
  const cur = useSift((state) => state.getSelected());
  return (
    <div className="w-full h-full flex flex-col items-center py-3 gap-2 px-2">
      {target_image && (
        <div className="w-full flex justify-start flex-none gap-1 items-center">
          <Label className="font-bold">Target Image</Label>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setTargetImage(null)}
          >
            Clear
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setImageExpanded(!imageExpanded)}
          >
            {imageExpanded ? "Collapse" : "Expand"}
          </Button>
        </div>
      )}
      <div className="w-full overflow-auto flex-grow justify-center border border-dashed mx-4">
        <ImageUpload
          onSetImage={setTargetImage}
          image={target_image}
          imgClassName={imageExpanded ? "object-cover" : "h-full m-auto"}
          // TODO: use object-contain to do this; however it doesn't seem to work
          className="w-full h-full object-cover rounded-lg shadow-md"
          content="Import Target Image"
        />
      </div>

      <div className="w-full flex-none">
        <TooltipProvider>
          <div className="flex flex-row items-center justify-around h-full gap-5">
            <LabelButton
              disabled={!cur}
              status="Not Match"
              hotkey={["h", "n", "1"]}
            />
            <LabelButton disabled={!cur} status="Keep" hotkey={["p", "2"]} />
            <LabelButton
              disabled={!cur}
              status="Match"
              hotkey={["l", "m", "3"]}
            />
          </div>
        </TooltipProvider>
      </div>
    </div>
  );
}
