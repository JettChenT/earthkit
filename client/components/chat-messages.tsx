import { ClientMessage } from "@/app/osm/actions";
import { useUIState } from "ai/rsc";

import { useEffect, useRef } from "react";
import MarkdownRenderer from "./markdown-render";
import {
  MatchDecorator,
  WidgetType,
  Decoration,
  DecorationSet,
  EditorView,
  ViewPlugin,
  keymap,
  ViewUpdate,
} from "@codemirror/view";
import "codemirror/lib/codemirror.css";
import CodeMirror from "@uiw/react-codemirror";
import { osm_placeholders, location_placeholders } from "./chatbox";

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
              {role === "assistant" ? (
                <div className="prose prose-stone text-sm pl-0">
                  <MarkdownRenderer content={content} />
                </div>
              ) : (
                <CodeMirror
                  value={content}
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
              )}
              {lowerIndicators}
            </div>
          </div>
        )
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}
