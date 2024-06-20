import { ReactNode } from "react";
import {
  StreamableValue,
  createAI,
  createStreamableUI,
  createStreamableValue,
  getMutableAIState,
} from "ai/rsc";
import { embed, streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { Pinecone, RecordMetadata } from "@pinecone-database/pinecone";
import { SYSTEM_PROMPT } from "@/lib/prompting";

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
  content: string;
  upperIndicator?: ReactNode; // used to display server progress
  lowerIndicators?: ReactNode[]; // used to display execution results
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

async function sendMessage(user_req: string, sys_results: string[] = []) {
  "use server";

  const history = getMutableAIState();
  const textStream = createStreamableValue<string>();
  const upperIndicatorStream = createStreamableUI();
  const progressStream = createStreamableValue<ProgressUpdate>();

  (async () => {
    const sysMessages = sys_results.map((result) => ({
      role: "system",
      content: result,
    }));

    history.update([
      ...history.get(),
      ...sysMessages,
      {
        role: "user",
        content: user_req,
      },
    ]);
    upperIndicatorStream.update(<div>Embedding...</div>);
    const embd_results = await embed({
      model: openai.embedding("text-embedding-3-small"),
      value: user_req,
    });
    upperIndicatorStream.update(<div>Querying...</div>);
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
    upperIndicatorStream.done(<></>);
    const { textStream: txtStream, text: resultText } = await streamText({
      model: openai("gpt-4o"),
      system: SYSTEM_PROMPT,
      messages: history.get(),
    });

    for await (const text of txtStream) {
      textStream.update(text);
    }
    const final_text = await resultText;
    history.done((messages: ServerMessage[]) => [
      ...messages,
      { role: "assistant", content: final_text },
    ]);
    progressStream.done({ kind: "done", value: final_text });
    textStream.done();
  })();
  return {
    textStream: textStream.value,
    upperIndicator: upperIndicatorStream.value,
    progressStream: progressStream.value,
  };
}

export const AI = createAI({
  initialAIState: [] as AIState,
  initialUIState: [] as UIState,
  actions: {
    sendMessage,
  },
});
