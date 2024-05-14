import { createSupabaseServerClient } from "@/utils/supabase/server";
import { headers } from "next/headers";
import Link from "next/link";
import { redirect } from "next/navigation";
import SignInGoogleButton from "../../../components/auth/signin-google-button";
import { SubmitButton } from "../../../components/auth/submit-button";

export default function Login({
  searchParams,
}: {
  searchParams: { message: string };
}) {
  const signUp = async (formData: FormData) => {
    "use server";

    const origin = headers().get("origin");
    const email = formData.get("email") as string;
    const password = formData.get("password") as string;
    const supabase = createSupabaseServerClient();

    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${origin}/auth/callback`,
      },
    });

    if (error) {
      console.log(error);

      return redirect(`/auth/signup?message=${error.message}`);
    }

    return redirect(
      "/auth/login?message=Check email to continue sign in process"
    );
  };

  return (
    <div className="mx-auto my-20 flex-1 flex flex-col w-full px-8 sm:max-w-md justify-center gap-2">
      <Link
        href="/"
        className="absolute left-20 top-8 py-2 px-4 rounded-md no-underline text-foreground bg-btn-background hover:bg-btn-background-hover flex items-center group text-sm"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="mr-2 h-4 w-4 transition-transform group-hover:-translate-x-1"
        >
          <polyline points="15 18 9 12 15 6" />
        </svg>{" "}
        Back
      </Link>

      <form className="animate-in flex-1 flex flex-col w-full justify-center gap-2 text-foreground">
        <label className="text-md" htmlFor="email">
          Email
        </label>
        <input
          className="rounded-md px-4 py-2 bg-inherit border mb-6"
          name="email"
          placeholder="you@example.com"
          required
        />
        <label className="text-md" htmlFor="password">
          Password
        </label>
        <input
          className="rounded-md px-4 py-2 bg-inherit border mb-6"
          type="password"
          name="password"
          placeholder="••••••••"
          required
        />
        <SubmitButton
          formAction={signUp}
          className="bg-purple-900 hover:bg-purple-950 rounded-md px-4 py-2 text-foreground mb-2"
          pendingText="Signing Up..."
        >
          Sign Up
        </SubmitButton>
        <div>
          <p className="mb-2">Already joined?</p>
          <a href="/auth/login" className="hover:text-inherit">
            <div className="justify-center items-center border-spacing-1 text-center rounded-md bg-green-700 hover:bg-green-800 px-4 py-2 mb-2">
              LOGIN NOW
            </div>
          </a>
        </div>
        <div id="authProvider" className="flex flex-col justify-center gap-5">
          <p className="flex justify-center">
            Or login / signup with these external providers
          </p>
          <SignInGoogleButton nextUrl="/"></SignInGoogleButton>
        </div>
      </form>
      {searchParams?.message && (
        <p className="mt-4 text-red-400 p-4 bg-foreground/10 text-foreground text-center">
          {searchParams.message}
        </p>
      )}
    </div>
  );
}
