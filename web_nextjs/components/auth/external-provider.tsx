import SignInGoogleButton from "./signin-google-button";

export const ExternalProvider = () => {
  return (
    <div id="authProvider" className="flex flex-col justify-center gap-5">
      <p className="flex justify-center">
        Or login / signup with these external providers
      </p>
      <SignInGoogleButton nextUrl="/"></SignInGoogleButton>
    </div>
  );
};
