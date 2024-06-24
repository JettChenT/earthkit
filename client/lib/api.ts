"use client";
import ky from "ky";
import { API_URL } from "./constants";
import { getHeaders } from "./supabase/client";

export async function getKy() {
  let kyInst = ky.create({
    prefixUrl: API_URL,
    ...(await getHeaders()),
  });
  return kyInst;
}

export async function fetcher<T>(url: string) {
  const kyInst = await getKy();
  const res = await kyInst.get(url).json<T>();
  return res;
}
