"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useChat } from "ai/react";

export default function OSM() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "api/osmnl",
  });
  return (
    <div className="w-full h-full">
      <div className="w-1/2">
        {messages.map((message) => (
          <div key={message.id}>
            {message.role}: {message.content}
            {message.toolInvocations?.map((tool) => (
              <div key={tool.toolCallId}>
                {tool.toolName}: {JSON.stringify(tool.args)}
              </div>
            ))}
          </div>
        ))}
        <form onSubmit={handleSubmit}>
          <Input type="text" value={input} onChange={handleInputChange} />
          <Button type="submit">Submit</Button>
        </form>
      </div>
    </div>
  );
}
