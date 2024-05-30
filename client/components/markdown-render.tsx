import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { docco } from "react-syntax-highlighter/dist/esm/styles/hljs";
import js from "react-syntax-highlighter/dist/esm/languages/hljs/javascript";
import overpassTurbo from "@/lib/osm_lang";
import { useEffect, useRef } from "react";

SyntaxHighlighter.registerLanguage("overpass", overpassTurbo);

interface MarkdownRendererInterface {
  content: string;
}

export default function MarkdownRenderer({
  content,
}: MarkdownRendererInterface) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) => (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="text-blue-700 hover:underline"
          >
            {children}
          </a>
        ),
        ul: ({ children }) => <ul className="list-disc">{children}</ul>,
        ol: ({ children }) => <ul className="list-decimal">{children}</ul>,
        li: ({ children }) => <li> {children}</li>,
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");
          return !inline && match ? (
            <SyntaxHighlighter
              {...props}
              style={docco}
              language={"overpass"}
              PreTag="div"
            >
              {String(children).replace(/\n$/, "")}
            </SyntaxHighlighter>
          ) : (
            <code {...props} className={className}>
              {children}
            </code>
          );
        },
      }}
      className="ml-1"
    >
      {content}
    </ReactMarkdown>
  );
}
