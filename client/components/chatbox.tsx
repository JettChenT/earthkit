import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { CornerDownLeft } from "lucide-react";
import { MentionsInput, Mention, OnChangeHandlerFunc } from "react-mentions";
import { OSMOrama, searchDb } from "@/app/osm/searchSuggestions";

interface ChatboxProps {
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  handleInputChange: OnChangeHandlerFunc;
  input: string;
  db: OSMOrama | null;
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
      <div className="flex-1">
        <MentionsInput
          className="w-full h-10 resize-none border-0 focus:outline-none focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
          placeholder="Type a message... (@to mention)"
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          allowSuggestionsAboveCursor={true}
        >
          <Mention
            trigger="@"
            renderSuggestion={(suggestion) => (
              <div className="bg-grey-300">{suggestion.display}</div>
            )}
            style={{
              backgroundColor: "blue",
            }}
            data={(query, callback) => {
              console.log(query);
              if (db) {
                searchDb(db, query).then((results) => {
                  callback(
                    results.hits.map((hit) => ({
                      id: hit.id,
                      display: hit.document.name,
                    }))
                  );
                });
              }
            }}
          />
          {/* <Mention trigger="@" data={[{ id: "1", display: "test" }]} /> */}
        </MentionsInput>
      </div>
      <div className="flex items-center justify-between">
        <div className="text-xs text-gray-500 ml-4">@ Mention</div>
        <Button type="submit" variant="secondary" className="py-0" size={"sm"}>
          <CornerDownLeft className="size-3 font-bold h-3 w-3" />
        </Button>
      </div>
    </form>
  );
}
