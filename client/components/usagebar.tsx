"use client";

import useClerkSWR from "@/lib/api";
import { Skeleton } from "./ui/skeleton";
import { Progress } from "./ui/progress";
import { UsageData } from "@/lib/db";
import { useAuth } from "@clerk/nextjs";

export function UsageBar() {
  let { isSignedIn } = useAuth();
  if (!isSignedIn) return null;
  let { data, isLoading } = useClerkSWR("/api/usage");
  data = data as UsageData;
  if (isLoading) return <Skeleton className="h-2 w-full" />;
  return (
    <div className="flex flex-col gap-1 px-6">
      <div className="text-sm">{data.remaining} Usage Units Left</div>
      <Progress
        className="h-[10px]"
        value={(data.remaining / data.quota) * 100}
      />
    </div>
  );
}
