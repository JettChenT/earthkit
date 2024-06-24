"use client";
import ky from "ky";
import { API_URL } from "./constants";
import useSWR from "swr";
import { useAuth } from "@clerk/nextjs";

export function useKy() {
  const getKyInst = async () => {
    const { getToken } = useAuth();
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
  const getKyInst = useKy();
  const fetcher = async (...args: [RequestInfo]) => {
    const kyInst = await getKyInst();
    return kyInst.get(url).json();
  };

  return useSWR(url, fetcher);
}
