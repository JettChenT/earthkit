import { Redis } from "@upstash/redis";

const DEFAULT_CREDIT = 500;

export interface UsageData {
  quota: number;
  remaining: number;
}

export async function fetchUsage(
  redis: Redis,
  user_id: string
): Promise<UsageData> {
  const fields = await redis.hmget(user_id, "remaining", "quota");
  if (fields === null) {
    await redis.hmset(user_id, {
      remaining: DEFAULT_CREDIT,
      quota: DEFAULT_CREDIT,
    });
    return { quota: DEFAULT_CREDIT, remaining: DEFAULT_CREDIT };
  }
  return {
    remaining: fields.remaining as number,
    quota: fields.quota as number,
  };
}

export async function verifyCost(
  redis: Redis,
  user_id: string,
  cost: number
): Promise<number> {
  if (cost < 0) throw new Error("Cost must be non-negative");
  let remaining = await redis.hincrby(user_id, "remaining", -cost);
  if (remaining == -cost && (await redis.hget(user_id, "quota")) === null) {
    await redis.hmset(user_id, {
      remaining: DEFAULT_CREDIT - cost,
      quota: DEFAULT_CREDIT,
    });
    remaining = DEFAULT_CREDIT - cost;
  }
  if (remaining < 0) {
    await redis.hincrby(user_id, "remaining", cost);
    throw new Error(
      `Quota exceeded. This operation requires a credit of at least ${cost}. You have ${
        remaining + cost
      } credits remaining.`
    );
  }
  return remaining;
}
