import { createSupabaseServerClient } from "@/utils/supabase/server";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);

  const supabase = createSupabaseServerClient();

  const { error } = await supabase.auth.signOut();

  if (!error) {
    return NextResponse.redirect(`${origin}`);
  }

  return NextResponse.redirect(`${origin}/auth/error`);
}
