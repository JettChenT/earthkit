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

const Kbd = ({ children }: { children: React.ReactNode }) => {
  return (
    <span className="inline-block shadow-sm bg-gray-100 text-gray-700 text-xs font-mono font-semibold px-2 py-1 rounded">
      {children}
    </span>
  );
};

const LabelButton = ({
  status,
  hotkey,
  explainer,
}: {
  status: LabelType;
  hotkey?: string[];
  explainer?: string;
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
  const cur = useSift((state) => state.getSelected());
  return (
    <div className="w-full h-full flex flex-col">
      <div className="my-5 p-3 mx-auto">
        <ImageUpload
          onSetImage={setTargetImage}
          image={target_image}
          imgClassName="max-h-28"
          className="px-20"
          content="Import Reference Image"
        />
      </div>
      <div>
        <TooltipProvider>
          <div className="flex flex-row items-center justify-center h-full gap-5">
            <LabelButton status="Match" hotkey={["h", "m", "1"]} />
            <LabelButton status="Keep" hotkey={["p", "2"]} />
            <LabelButton status="Not Match" hotkey={["l", "n", "3"]} />
          </div>
        </TooltipProvider>
      </div>
    </div>
  );
}
