import { NextRequest, NextResponse } from "next/server";
import { Redis } from "@upstash/redis";
import { fetchUsage } from "@/lib/db";
import { auth } from "@clerk/nextjs/server";

const redis = Redis.fromEnv();

export async function GET() {
  try {
    const { userId } = auth();
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const usageData = await fetchUsage(redis, userId);
    return NextResponse.json(usageData);
  } catch (error) {
    return NextResponse.json({ error }, { status: 500 });
  }
}
