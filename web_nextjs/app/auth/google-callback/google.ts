import { createClient } from "@/utils/supabase/server";

async function handleSignInWithGoogle(response: { credential: any }) {
  const supabase = createClient();
  const { data, error } = await supabase.auth.signInWithIdToken({
    provider: "google",
    token: response.credential,
  });
}

export default handleSignInWithGoogle;
