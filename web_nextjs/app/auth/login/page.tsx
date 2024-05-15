import { createSupabaseServerClient } from "@/utils/supabase/server";
import Link from "next/link";
import { redirect } from "next/navigation";
import SignInGoogleButton from "../../../components/auth/signin-google-button";
import { FormUserNamePassword } from "@/components/auth/form-username-password";
import { ExternalProvider } from "@/components/auth/external-provider";

export default function Login({
  searchParams,
}: {
  searchParams: { message: string };
}) {
  const login = async (formData: FormData) => {
    "use server";

    const email = formData.get("email") as string;
    const password = formData.get("password") as string;
    const supabase = createSupabaseServerClient();

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      return redirect("/auth/login?message=Could not authenticate user");
    }

    return redirect("/");
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
        <h1 className="flex justify-center">Login to your account</h1>
      </div>
      <FormUserNamePassword
        formAction={login}
        pendingText="Login..."
        className="bg-green-700 hover:bg-green-800 rounded-md px-4 py-2 text-foreground mb-2"
        buttonText="Login"
      ></FormUserNamePassword>
      <ExternalProvider></ExternalProvider>

      <div className="flex flex-row justify-center mt-2">
        <p className="flex-1 self-center text-center">Want to join?</p>
        <a
          href="/auth/signup"
          className="hover:text-inherit flex-1 self-center block mx-auto my-2"
        >
          <div className="border-spacing-1 text-center rounded-md bg-purple-800 hover:bg-purple-900 py-2 mx-auto px-10">
            Start for free
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
