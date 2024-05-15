import { ComponentProps } from "react";
import { SubmitButton } from "./submit-button";

type Props = ComponentProps<"button"> & {
  pendingText?: string;
  buttonText?: string;
};
export const FormUserNamePassword = (props: Props) => {
  return (
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
        formAction={props.formAction}
        className={props.className}
        pendingText={props.pendingText}
      >
        {props.buttonText}
      </SubmitButton>
    </form>
  );
};
