import { ReactNode } from "react";
import {
  StreamableValue,
  createAI,
  createStreamableValue,
  getMutableAIState,
  streamUI,
} from "ai/rsc";
import { embed } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { Pinecone, RecordMetadata } from "@pinecone-database/pinecone";
import { SYSTEM_PROMPT } from "@/lib/prompting";
import { nanoid } from "nanoid";

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_API_BASE,
});

const pc = new Pinecone();
const index = pc.index("osm-queries");

export interface ServerMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ClientMessage {
  id: string;
  role: "user" | "assistant";
  display: ReactNode;
}

export interface ProgressUpdate {
  kind: "progress" | "done";
  value: string;
}

export interface GenerationResponse {
  clientMessage: ClientMessage;
  progress: StreamableValue<ProgressUpdate>;
}

export type AIState = Array<ServerMessage>;

export type UIState = Array<ClientMessage>;

async function sendMessage(
  user_req: string
): Promise<[ClientMessage, StreamableValue<ProgressUpdate>]> {
  "use server";

  const history = getMutableAIState();

  console.log(user_req);
  console.log("embedding...");
  history.update([
    ...history.get(),
    {
      role: "user",
      content: user_req,
    },
  ]);
  const embd_results = await embed({
    model: openai.embedding("text-embedding-3-small"),
    value: user_req,
  });
  console.log("querying...");
  const query_results = await index.query({
    topK: 5,
    vector: embd_results.embedding,
    includeMetadata: true,
  });

  const fmt_query = (x: RecordMetadata) => {
    return `Natural Language Query: ${x.nl}\nOverpass Turbo Query: ${x.query}`;
  };
  history.update([
    ...history.get(),
    {
      role: "system",
      content: `Past generations for reference: ${query_results.matches
        .map((r) => fmt_query(r.metadata!))
        .join("\n-------\n")}`,
    },
  ]);
  console.log("streaming...");

  const statusStream = createStreamableValue<ProgressUpdate>();

  const result = await streamUI({
    model: openai("gpt-4o"),
    system: SYSTEM_PROMPT,
    messages: history.get(),
    text: ({ content, done }) => {
      if (done) {
        statusStream.done({ kind: "done", value: content });
        history.done((messages: ServerMessage[]) => [
          ...messages,
          { role: "assistant", content },
        ]);
      }

      return content;
    },
  });

  return [
    {
      id: nanoid(),
      role: "assistant",
      display: result.value,
    },
    statusStream.value,
  ];
}

export const AI = createAI({
  initialAIState: [] as AIState,
  initialUIState: [] as UIState,
  actions: {
    sendMessage,
  },
});
