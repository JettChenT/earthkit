"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";

import { createClient } from "@/lib/supabase/server";
import { Provider } from "@supabase/supabase-js";
import { IS_LOCAL } from "@/lib/constants";

export async function login(provider: Provider) {
  const supabase = createClient();
  const redirectTo = IS_LOCAL
    ? "http://localhost:3000/auth/callback"
    : "https://earthkit.app/auth/callback";
  console.log("is local", IS_LOCAL, redirectTo);
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider,
    options: {
      redirectTo,
    },
  });
  console.log("data", data);
  if (data.url) {
    redirect(data.url);
  }
  revalidatePath("/", "layout");
}
