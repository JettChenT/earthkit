import { put } from "@vercel/blob";
import { NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";

export async function POST(request: Request): Promise<NextResponse> {
  if (!request.body) {
    return NextResponse.json({ error: "No body" }, { status: 400 });
  }

  const { searchParams } = new URL(request.url);
  const type = searchParams.get("type");

  if (!type || !["JPG", "JPEG", "PNG", "GIF"].includes(type.toUpperCase())) {
    return NextResponse.json(
      { error: "Invalid or missing file type" },
      { status: 400 }
    );
  }

  const extension = type.toLowerCase();
  const filename = `earthkit_uploads/${uuidv4()}.${extension.toLowerCase()}`;

  const blob = await put(filename, request.body, {
    access: "public",
  });

  return NextResponse.json(blob);
}
