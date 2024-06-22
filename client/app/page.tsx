import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ArrowRight } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="container max-w-xl mx-auto my-10 flex flex-col items-start">
      <header className="flex justify-between w-full items-center drop-shadow-sm bg-gray-50 border-gray-100 text-gray-700 rounded-xl py-2 px-4">
        <h1 className="font-bold text-3xl font-mono drop-shadow-sm">
          <span className="text-blue-700">E</span>
          <span className="text-green-700">K</span>
        </h1>
        <div className="space-x-4 flex items-center">
          <Link
            href="https://discord.com"
            className="underline text-sm variant-ghost"
          >
            Discord
          </Link>
          <Link
            href="https://github.com"
            className="underline text-sm variant-ghost"
          >
            GitHub
          </Link>
          <Button asChild size="sm">
            <Link href="/sift" className="font-bold font-mono">
              App
            </Link>
          </Button>
        </div>
      </header>
      <article className="prose prose-neutral mt-10">
        <h2 className="font-mono">
          <span className="text-blue-700">E</span>
          <span>arth</span>
          <span className="text-green-700">K</span>
          <span>it</span>
        </h2>
        <p className="font-bold">
          A Nifty and Smart toolkit that helps you Geolocate faster with AI.
        </p>
        <p>
          Automate the process of constructing overpass turbo queries, sifting
          through streetview/satellite images, initial geo-guessing, and more.
        </p>
        <iframe
          className="w-full h-[300px] rounded-md mb-5"
          src="https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?si=m3zu3VYFTJFIIEkh&amp;rel=0&amp;modestbranding=1"
          title="YouTube video player"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          referrerPolicy="strict-origin-when-cross-origin"
          allowFullScreen
        ></iframe>
        <Button asChild>
          <Link href="/sift" className="font-mono font-extrabold no-underline">
            Take me to the App <ArrowRight className="size-4 ml-2" />
          </Link>
        </Button>
      </article>
    </div>
  );
}
