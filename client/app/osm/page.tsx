"use client";

import { Chatbox } from "@/components/chatbox";
import { useChat } from "ai/react";

import { ChatMessages } from "@/components/chat-messages";

export default function OSM() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: "api/osmnl",
  });
  return (
    <div className="w-full h-full">
      <div className="w-1/2">
        <ChatMessages messages={messages} />
        <Chatbox
          handleSubmit={handleSubmit}
          handleInputChange={handleInputChange}
          input={input}
        />
      </div>
    </div>
  );
}
