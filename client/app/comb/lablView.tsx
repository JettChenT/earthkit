import { Button } from "@/components/ui/button";
import { LabelType, useComb } from "../../lib/combStore";
import { useHotkeys, Keys } from "react-hotkeys-hook";
import { useRef } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import React from "react";

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
  const { setIdxData } = useComb();
  const btnRef = useRef<HTMLButtonElement>(null);
  useHotkeys(hotkey ?? "", () => {
    btnRef.current?.click();
  });

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            className="py-8 w-32"
            onClick={() => setIdxData({ status })}
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
  const { setIdxData } = useComb();
  const cur = useComb((state) => state.getSelected());
  return (
    <div className="w-full h-full">
      <TooltipProvider>
        <div className="flex flex-row items-center justify-center h-full gap-5">
          <LabelButton status="Match" hotkey={["h", "m", "1"]} />
          <LabelButton status="Keep" hotkey={["p", "2"]} />
          <LabelButton status="Not Match" hotkey={["l", "n", "3"]} />
        </div>
      </TooltipProvider>
    </div>
  );
}
