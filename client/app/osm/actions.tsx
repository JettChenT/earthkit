import { ReactNode } from "react";
import {
  StreamableValue,
  createAI,
  createStreamableUI,
  createStreamableValue,
  getMutableAIState,
} from "ai/rsc";
import {
  AssistantContent,
  CoreMessage,
  ImagePart,
  UserContent,
  embed,
  streamText,
} from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { Pinecone, RecordMetadata } from "@pinecone-database/pinecone";
import { SYSTEM_PROMPT } from "@/lib/prompting";
import { auth } from "@clerk/nextjs/server";
import { Redis } from "@upstash/redis";
import { verifyCost } from "@/lib/db";
import { Ratelimit } from "@upstash/ratelimit";
import { verify } from "crypto";
import { headers } from "next/headers";

const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(3, "1 h"), // Rate limit of 3 LM calls per hour
  prefix: "osm-anon",
});

const redis = Redis.fromEnv();

export type AIContent = UserContent | AssistantContent;

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_API_BASE,
});

const pc = new Pinecone();
const index = pc.index("osm-queries");

export interface ServerMessage {
  role: "user" | "assistant" | "system";
  content: AIContent;
}

export interface ClientMessage {
  id: string;
  role: "user" | "assistant";
  content: AIContent;
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

export type Model = "gpt-3.5-turbo" | "gpt-4o";

async function sendMessage(
  user_req: string,
  sys_results: string[] = [],
  images: string[] = [],
  model: Model = "gpt-3.5-turbo"
) {
  "use server";

  const history = getMutableAIState<typeof AI>();
  const textStream = createStreamableValue<string>();
  const upperIndicatorStream = createStreamableUI();
  const progressStream = createStreamableValue<ProgressUpdate>();

  console.log("images", images.length);
  const { userId } = auth();

  // if (!["gpt-3.5-turbo", "gpt-4o"].includes(model)) {
  //   throw new Error(`Invalid model: ${model}`);
  // }
  // if (!userId && model === "gpt-4o") {
  //   throw new Error("Unauthorized");
  // }
  // if (model === "gpt-4o") await verifyCost(redis, userId!, 1);

  (async () => {
    if (model === "gpt-4o") await verifyCost(redis, userId!, 1);
    const sysMessages = sys_results.map((result) => ({
      role: "system" as const,
      content: result,
    }));

    history.update([
      ...history.get(),
      ...sysMessages,
      {
        role: "user",
        content: [
          ...images.map(
            (image) =>
              ({
                type: "image",
                image,
              } as ImagePart)
          ),
          { type: "text", text: user_req },
        ],
      },
    ]);
    upperIndicatorStream.update(<div>Validating...</div>);
    try {
      if (userId) {
        await verifyCost(redis, userId, 1);
      } else {
        const ip =
          headers().get("x-forwarded-for") ||
          headers().get("x-real-ip") ||
          headers().get("cf-connecting-ip") ||
          "unknown";
        let res = await ratelimit.limit(`anon-${ip}`);
        if (!res.success) {
          throw new Error(
            "Rate limit exceeded. Please create an account to remove the limit."
          );
        }
      }
    } catch (e) {
      upperIndicatorStream.done(
        <div className="text-red-500">
          {e instanceof Error ? e.message : "Rate limit exceeded"}
        </div>
      );
      progressStream.done({ kind: "done", value: "Rate limit exceeded" });
      textStream.done();
      return;
    }

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
    console.log("history", history.get());
    const { textStream: txtStream, text: resultText } = await streamText({
      model: openai(model),
      system: SYSTEM_PROMPT,
      messages: history.get().map((x) => ({
        role: x.role,
        content: x.content as any,
      })),
    });

    for await (const text of txtStream) {
      textStream.update(text);
    }
    const final_text = await resultText;
    history.done([
      ...history.get(),
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
