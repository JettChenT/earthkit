import Navbar from "@/components/navbar";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardTitle,
} from "@/components/ui/card";
import { ArrowRight } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="container max-w-3xl mx-auto my-10 flex flex-col items-start">
      <Navbar />
      <article className="prose prose-neutral mt-10">
        <h2 className="font-mono">
          <span className="text-blue-700">E</span>
          <span>arth</span>
          <span className="text-green-700">K</span>
          <span>it</span>
        </h2>
        <Button className="not-prose" variant="link" asChild>
          <Link
            href="https://agent.earthkit.app"
            className="font-mono font-bold flex flex-row items-center bg-muted/70"
          >
            New: Introducing EarthKit Agent
            <ArrowRight className="size-4 ml-2" />
          </Link>
        </Button>
        <p className="font-bold">
          A Nifty toolkit that helps you Geolocate faster with AI.
        </p>
        <p>
          Automate the process of constructing overpass turbo queries, sifting
          through streetview/satellite images, initial geo-guessing, and more.
        </p>
        <p>
          EarthKit is{" "}
          <a href="https://github.com/JettChenT/earthkit">open-source</a> and
          will be self-hostable. Feel free to try it out here! A free compute
          credit will be granted on signup. Note that we are still in the early
          stages of development, and features may be slow, unstable, or
          unavailable.
        </p>
        <Button asChild>
          <Link href="/sift" className="font-mono font-extrabold no-underline">
            Take me to the App <ArrowRight className="size-4 ml-2" />
          </Link>
        </Button>
        <h2 className="font-bold text-lg mt-10">EarthKit Features:</h2>
        <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-2 justify-between mb-4">
          <FeatCard
            title="Sift Through Coordinates with Ease"
            description="Browse and annotate coordinates, streetview, satellite, and more"
            src="AcRr-K_J-xY"
          />
          <FeatCard
            title="OpenStreetmap Querying"
            description="Construct Overpass-Turbo Queries with Natural Langauge and Intelligent Suggestions"
            src="LkTQZiy7CX4"
          />
          <FeatCard
            title="Use AI to Accelerate Your Investigation"
            description="Enrich, sort, and filter your geographical data with SOTA models such as GPT-4o."
            src="J_pja8AdjSc"
          />
          <FeatCard
            title="Start Your Investigation with GeoEstimation"
            description="Estimate the location of any image with GeoCLIP"
            src="GaS5mnHb0Z8"
          />
        </div>
      </article>
    </div>
  );
}

function FeatCard({
  title,
  description,
  src,
}: {
  title: string;
  description: string;
  src: string;
}) {
  return (
    <Card>
      <CardContent className="px-4 p-2 bg-muted/20">
        <CardTitle className="text-lg mt-1">{title}</CardTitle>
        <CardDescription className="text-sm">{description}</CardDescription>
        <iframe
          className="w-full h-[200px] rounded-md mb-2"
          src={`https://www.youtube-nocookie.com/embed/${src}?si=m3zu3VYFTJFIIEkh&amp;rel=0&amp;modestbranding=1`}
          title="YouTube video player"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          referrerPolicy="strict-origin-when-cross-origin"
          allowFullScreen
        ></iframe>
      </CardContent>
    </Card>
  );
}
