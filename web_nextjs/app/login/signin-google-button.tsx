"use client";
import { createClient } from "@/utils/supabase/client";

const SignInGoogleButton = async (props: { nextUrl?: string }) => {
  const supabase = createClient();

  const handleLogin = async () => {
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${location.origin}/auth/callback?next=${
          props.nextUrl || ""
        }`,
      },
    });
  };

  return <button onClick={handleLogin}>Login with Google</button>;
};
export default SignInGoogleButton;
