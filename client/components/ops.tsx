"use client";
import { useStore } from "@/lib/store";

export default function Operations() {
  let { tool } = useStore();
  return (
    <div className="w-56 p-3 bg-opacity-80 absolute right-3 mt-3 bg-white rounded-md">
      <div className="text-primary">{tool}</div>
      <div>Here are what you need to do</div>
    </div>
  );
}
