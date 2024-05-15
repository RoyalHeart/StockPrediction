import { createSupabaseServerClient } from "@/utils/supabase/server";
import { headers } from "next/headers";
import Link from "next/link";
import { redirect } from "next/navigation";
import SignInGoogleButton from "../../../components/auth/signin-google-button";
import { SubmitButton } from "../../../components/auth/submit-button";
import { FormUserNamePassword } from "@/components/auth/form-username-password";
import { ExternalProvider } from "@/components/auth/external-provider";

export default function Login({
  searchParams,
}: {
  searchParams: { message: string };
}) {
  const signup = async (formData: FormData) => {
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
    <div className="mx-auto my-24 flex-1 flex flex-col w-full px-8 sm:max-w-md justify-center gap-2">
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
      <div className="text-2xl/loose mr-5 bg-gradient-to-r from-green-400 from-25% to-purple-500 to-80% block text-transparent bg-clip-text">
        <h1 className="flex justify-center">Start joining for free</h1>
      </div>
      <FormUserNamePassword
        formAction={signup}
        pendingText="Login..."
        className="justify-center items-center border-spacing-1 text-center rounded-md bg-purple-900 hover:bg-purple-950 px-4 py-2 mb-2"
        buttonText="Sign up"
      ></FormUserNamePassword>
      <ExternalProvider></ExternalProvider>
      <div className="flex flex-row justify-center mt-2">
        <p className="flex-1 self-center text-center">Already joined?</p>
        <a
          href="/auth/login"
          className="hover:text-inherit flex-1 self-center block mx-auto my-2"
        >
          <div className="border-spacing-1 text-center rounded-md bg-green-700 hover:bg-green-800 py-2 mx-auto px-10">
            Login now
          </div>
        </a>
      </div>
      {searchParams?.message && (
        <p className="mt-4 text-red-400 p-4 bg-foreground/10 text-foreground text-center">
          {searchParams.message}
        </p>
      )}
    </div>
  );
}
