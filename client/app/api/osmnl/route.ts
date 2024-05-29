import { type CoreMessage, streamText, embed, tool } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { Pinecone, RecordMetadata } from "@pinecone-database/pinecone";
import { z } from "zod";

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
    system:
      "You are a helpful bot that composes Overpass Turbo queries based on a natural language query from user. In your response, include the Overpass Turbo query in a code block with language set to `overpassql`. e.g. ```overpassql\n[out:json]\n(area[name=" +
      '"Dublin, Ireland"];\n->.a;\narea.a[' +
      'name="Dublin, Ireland"];\nout;\n```. Generally, structure your response by beginning with a brief description of what you are going to query, followed by a codeblock. DO NOT include anything after that.',
    messages,
  });

  return result.toAIStreamResponse();
}
