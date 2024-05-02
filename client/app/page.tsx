"use client";

import MapDisplay from "@/components/map";
import Sidebar from "@/components/sidebar";
import ResourceBar from "@/components/resource_bar";
import Image from "next/image";

export default function Home() {
  return (
    <main className="h-screen w-screen flex overflow-hidden">
      <Sidebar />
      <MapDisplay />
      <ResourceBar />
    </main>
  );
}
