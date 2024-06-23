import { AIContent, ClientMessage } from "@/app/osm/actions";
import { useUIState } from "ai/rsc";

import { EditorView } from "@codemirror/view";
import CodeMirror from "@uiw/react-codemirror";
import "codemirror/lib/codemirror.css";
import { useEffect, useRef } from "react";
import MarkdownRenderer from "../../components/markdown-render";
import { location_placeholders, osm_placeholders } from "./cm_common";
import { ImagePart, TextPart, ToolCallPart } from "ai";

function RenderContent({
  content,
  role,
}: {
  content: AIContent;
  role: "user" | "assistant";
}) {
  const parts: Array<TextPart | ImagePart | ToolCallPart> = Array.isArray(
    content
  )
    ? content
    : [
        {
          type: "text",
          text: content,
        },
      ];
  return (
    <div className="flex flex-col gap-2 mb-2">
      {parts.map((part, idx) => {
        switch (part.type) {
          case "image":
            if (typeof part.image === "string" || part.image instanceof URL) {
              return <img key={idx} src={part.image.toString()} alt="image" />;
            } else {
              return <span key={idx}>Image type not supported yet</span>;
            }
          case "text":
            if (role === "assistant") {
              return (
                <div className="prose prose-stone text-sm pl-0" key={idx}>
                  <MarkdownRenderer content={part.text} />
                </div>
              );
            }
            return (
              <CodeMirror
                key={idx}
                value={part.text}
                theme={EditorView.theme({
                  "&": {
                    backgroundColor: "transparent",
                  },
                })}
                extensions={[
                  EditorView.lineWrapping,
                  osm_placeholders,
                  location_placeholders,
                ]}
                editable={false}
                basicSetup={{
                  lineNumbers: false,
                  foldGutter: false,
                  highlightActiveLine: false,
                }}
              />
            );
          case "tool-call":
            return <div key={idx}>{part.toolName}</div>;
        }
      })}
    </div>
  );
}

export function ChatMessages() {
  const [messages, _] = useUIState();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "auto" });
    }
  }, [messages]);

  return (
    <div
      key="1"
      className="grow flex flex-col items-start gap-4 p-4 md:p-8 w-full lg:mx-auto overflow-y-auto"
    >
      {messages.map(
        ({
          id,
          role,
          content,
          upperIndicator,
          lowerIndicators,
        }: ClientMessage) => (
          <div key={id} className="flex items-start gap-4 w-full">
            <div
              className={`${
                role === "user"
                  ? "bg-gray-50 dark:bg-gray-950 dark:text-gray-50"
                  : "bg-transparent"
              } rounded-lg p-3 max-w-[80%] flex-1`}
            >
              <div
                className={`font-medium text-sm ${
                  role === "user" ? "" : "text-gray-500 dark:text-gray-400"
                }`}
              >
                {role === "user" ? "You" : "Assistant"}
              </div>
              {upperIndicator}
              <RenderContent content={content} role={role} />
              {lowerIndicators}
            </div>
          </div>
        )
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}
