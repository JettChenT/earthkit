import Navbar from "@/components/navbar";
import HeroVideoDialog from "@/components/magicui/heroVideoDialog";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { BotIcon, TelescopeIcon } from "lucide-react";

const FORM_URL =
  "https://airtable.com/appYaguuyo7lOdmlS/pagleXIu8wH7s0BxQ/form";

export default function AgentPage() {
  return (
    <div className="container max-w-3xl mx-auto my-10 flex flex-col items-start">
      <Navbar />
      <article className="prose prose-neutral mt-10">
        <h2 className="font-mono">
          EarthKit <span className="text-blue-800">Agent</span>
        </h2>
        <p className="font-bold">
          Multi-Modal Agent for Geolocation and Verification.
        </p>
        <p>
          EarthKit Agent combines the real-time information such as web search,
          streetview, satellite imagery with the power of ML geolocation models
          such as <a href="https://earthkit.app/geoclip">GeoCLIP</a> to
          accurately identify and verify the location of multi-media sources.
        </p>
      </article>
      <div className="w-full flex justify-start mt-4">
        <div className="relative rounded-2xl p-1 overflow-hidden">
          <HeroVideoDialog
            animationStyle="from-center"
            videoSrc="https://www.youtube.com/embed/qNTK1wd3SrU?si=yRyCizdWFNHiP-z5"
            thumbnailSrc="/AgentDemo2.png"
            thumbnailAlt="Agent Video"
          />
        </div>
      </div>
      <div className="mt-3 flex flex-row gap-2 items-center">
        <Button asChild>
          <Link target="_blank" href={FORM_URL}>
            <BotIcon className="size-4 mr-2" />
            Get Access
          </Link>
        </Button>
        <Button asChild variant={"outline"}>
          <Link target="_blank" href={"https://agent.earthkit.app/feed"}>
            <TelescopeIcon className="size-4 mr-2" />
            Explore Public Agent Sessions
          </Link>
        </Button>
      </div>
    </div>
  );
}
