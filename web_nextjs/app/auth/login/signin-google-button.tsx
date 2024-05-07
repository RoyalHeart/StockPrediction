"use client";
import { createSupabaseBrowserClient } from "@/utils/supabase/client";

const SignInGoogleButton = async (props: { nextUrl?: string }) => {
  const supabase = createSupabaseBrowserClient();
  const handleLogin = async () => {
    const redirectionRoute = `${location.origin}/auth/callback?next=${
      props.nextUrl || ""
    }`;
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: redirectionRoute,
      },
    });
  };

  return <button onClick={handleLogin}>Login with Google</button>;
};
export default SignInGoogleButton;
