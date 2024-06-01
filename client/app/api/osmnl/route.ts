import { type CoreMessage, streamText, embed, tool } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { Pinecone, RecordMetadata } from "@pinecone-database/pinecone";
import { z } from "zod";
import { SYSTEM_PROMPT } from "../../../lib/prompting";

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_API_BASE,
});

const pc = new Pinecone();
const index = pc.index("osm-queries");

export async function POST(req: Request) {
  const { messages }: { messages: CoreMessage[] } = await req.json();
  const user_req = messages.at(messages.length - 1)?.content;
  console.log(user_req);
  console.log("embedding...");
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
  messages.push({
    role: "system",
    content: `Past generations for reference: ${query_results.matches
      .map((r) => fmt_query(r.metadata!))
      .join("\n-------\n")}`,
  });
  console.log(messages[messages.length - 1].content);
  console.log("streaming...");
  const result = await streamText({
    model: openai("gpt-4o"),
    system: SYSTEM_PROMPT,
    messages,
  });

  return result.toAIStreamResponse();
}
