"use client";
import { useStore } from "@/lib/store";
import GeoCLIPPanel from "./operations/geoclip";

export default function Operations() {
  let { tool } = useStore();
  return (
    <div className="w-72 p-3 bg-opacity-80 absolute right-3 mt-3 bg-white rounded-md">
      {(() => {
        switch (tool) {
          case "geoclip":
            return <GeoCLIPPanel />;
          default:
            return <div>TODO</div>;
        }
      })()}
    </div>
  );
}
