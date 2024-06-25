"use client";
import ky from "ky";
import { API_URL } from "./constants";
import useSWR, { useSWRConfig } from "swr";
import { useAuth } from "@clerk/nextjs";

export function useKy() {
  const { getToken } = useAuth();
  const getKyInst = async () => {
    let kyInst = ky.create({
      prefixUrl: API_URL,
      headers: {
        Authorization: `Bearer ${await getToken()}`,
      },
    });
    return kyInst;
  };
  return getKyInst;
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
