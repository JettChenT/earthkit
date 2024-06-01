"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";

import { createClient } from "@/lib/supabase/server";

export async function login(formData: FormData) {
  const supabase = createClient();
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: "github",
    options: {
      redirectTo:
        process.env.NODE_ENV === "development"
          ? "http://localhost:3000/auth/callback"
          : "https://earthkit.app/auth/callback",
    },
  });
  if (data.url) {
    redirect(data.url);
  }
  revalidatePath("/", "layout");
}
