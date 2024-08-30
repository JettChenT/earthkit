"use client";
import ky from "ky";
import { API_URL, AI_API_URL } from "../constants";
import useSWR, { useSWRConfig } from "swr";
import { useAuth } from "@clerk/nextjs";
import createClient, { Middleware } from "openapi-fetch";
import type { paths } from "./schema";

export function useKy() {
  const { getToken } = useAuth();
  const getKyInst = async () => {
    const token = await getToken();
    let kyInst = ky.create({
      prefixUrl: API_URL,
      timeout: 1000 * 60 * 5, // 5 minutes
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    return kyInst;
  };
  return getKyInst;
}

export function useAPIClient(kind: "api" | "ai" = "api") {
  const { getToken } = useAuth();
  const baseUrl = kind === "api" ? API_URL : AI_API_URL;
  const getClient = async () => {
    const token = await getToken();
    console.debug("token", token);
    let client = createClient<paths>({
      baseUrl,
      headers: token !== null ? { Authorization: `Bearer ${token}` } : {},
    });
    return client;
  };
  return getClient;
}

export default function useClerkSWR(url: string) {
  const { getToken } = useAuth();

  const fetcher = async (...args: [RequestInfo]) => {
    return fetch(...args, {
      headers: { Authorization: `Bearer ${await getToken()}` },
    }).then((res) => res.json());
  };

  return useSWR(url, fetcher);
}
