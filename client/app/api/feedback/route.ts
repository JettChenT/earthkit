import { NextRequest, NextResponse } from "next/server";

const AIRTABLE_BASE_ID = process.env.AIRTABLE_BASE_ID!;
const AIRTABLE_TABLE_NAME = process.env.AIRTABLE_TABLE_NAME!;
const AIRTABLE_TOKEN = process.env.AIRTABLE_TOKEN!;

type FeedbackRequestBody = {
  feedback: string;
};

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const { feedback }: FeedbackRequestBody = await req.json();

  if (!feedback) {
    return NextResponse.json(
      { error: "Missing required field: feedback" },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(
      `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/${AIRTABLE_TABLE_NAME}`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${AIRTABLE_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fields: {
            feedback,
          },
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to create record: ${response.statusText}`);
    }

    const createdRecord = await response.json();
    return NextResponse.json({ success: true, record: createdRecord });
  } catch (error) {
    console.error("Error creating record in Airtable:", error);
    return NextResponse.json(
      { error: "Failed to create record" },
      { status: 500 }
    );
  }
}
