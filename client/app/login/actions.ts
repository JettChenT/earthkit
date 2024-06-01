"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";

import { createClient } from "@/lib/supabase/server";
import { Provider } from "@supabase/supabase-js";

export async function login(provider: Provider) {
  const supabase = createClient();
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider,
    options: {
      redirectTo: process.env.IS_LOCAL
        ? "http://localhost:3000/auth/callback"
        : "https://earthkit.app/auth/callback",
    },
  });
  if (data.url) {
    redirect(data.url);
  }
  revalidatePath("/", "layout");
}
