import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { CornerDownLeft, Tag } from "lucide-react";
import { MentionsInput, Mention, OnChangeHandlerFunc } from "react-mentions";
import { OSMOrama, searchDb, Document } from "@/app/osm/searchSuggestions";
import keyIcon from "@/public/icons/osm_element_key.svg";
import tagIcon from "@/public/icons/osm_element_tag.svg";
import Image from "next/image";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";

interface ChatboxProps {
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  handleInputChange: OnChangeHandlerFunc;
  input: string;
  db: OSMOrama | null;
}

function SuggestionItem({
  data,
  focused,
}: {
  data: Document;
  focused: boolean;
}) {
  return (
    <HoverCard open={focused} openDelay={0} closeDelay={0}>
      <HoverCardTrigger
        className={`px-1 flex items-center gap-2 rounded-sm ${
          focused ? "bg-gray-300" : ""
        }`}
        data-state={focused ? "open" : "closed"}
      >
        <Image
          src={data.type === "key" ? keyIcon : tagIcon}
          className="fill-orange-700"
          alt="key"
          width={16}
          height={16}
        />
        {data.name}
      </HoverCardTrigger>
      <HoverCardContent side="right" className="animate-none ml-0.5">
        <article className="flex flex-col gap-1">
          <div className="text-sm font-bold">{data.name}</div>
          <div className="text-xs text-gray-500">type: {data.type}</div>
          <div className="text-xs text-gray-500">{data.description}</div>
        </article>
      </HoverCardContent>
    </HoverCard>
  );
}

function DisplayItemInline({ data }: { data: Document }) {
  return <div>{data.name}</div>;
}

export function Chatbox({
  handleSubmit,
  handleInputChange,
  input,
  db,
}: ChatboxProps) {
  const handleKeyDown = (
    event:
      | React.KeyboardEvent<HTMLTextAreaElement>
      | React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event as unknown as React.FormEvent<HTMLFormElement>);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex-none p-2 bg-white border rounded-md w-full mb-3"
    >
      <div className="flex-1 pl-2">
        <MentionsInput
          className="w-full h-10 resize-none border-0 focus:outline-none focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
          placeholder="Type a message... (@to mention)"
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          allowSuggestionsAboveCursor={true}
          forceSuggestionsAboveCursor={true}
          style={{
            control: {
              fontSize: 14,
              fontWeight: "normal",
            },
            input: {
              outline: "none",
              boxShadow: "none",
            },
            suggestions: {
              list: {
                borderRadius: "5px",
                border: "1px solid rgba(0,0,0,0.1)",
                padding: "2px",
              },
            },
          }}
        >
          <Mention
            trigger="@"
            className="inline-block rounded-sm font-mono font-bold px-1 -mx-0.5 bg-yellow-200"
            displayTransform={(id, display) => {
              const data: Document = JSON.parse(display);
              const symbol = data.type === "key" ? "🔑" : "🏷️";
              return `${symbol} ${data.name}`;
            }}
            renderSuggestion={(
              entry,
              search,
              highlightedDisplay,
              idx,
              focused
            ) => {
              if (!entry.display) return;
              const data: Document = JSON.parse(entry.display);
              return <SuggestionItem data={data} focused={focused} />;
            }}
            data={(query, callback) => {
              if (db) {
                searchDb(db, query).then((results) => {
                  callback(
                    results.hits.map((hit) => ({
                      id: hit.id,
                      display: JSON.stringify(hit.document),
                    }))
                  );
                });
              }
            }}
          />
        </MentionsInput>
      </div>
      <div className="flex items-center justify-between">
        <div className="text-xs text-gray-500 ml-2">@ Mention</div>
        <Button type="submit" variant="secondary" className="py-0" size={"sm"}>
          <CornerDownLeft className="size-3 font-bold h-3 w-3" />
        </Button>
      </div>
    </form>
  );
}
