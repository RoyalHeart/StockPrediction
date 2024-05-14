import { createSupabaseServerClient } from "@/utils/supabase/server";
import LogoutIcon from "@mui/icons-material/Logout";
export const NavActions = async () => {
  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase.auth.getUser();
  let isUserLogined: boolean = true;
  if (error || !data?.user) {
    isUserLogined = false;
  }
  const userDisplayName = data.user?.identities?.at(0)?.identity_data?.name;
  const userEmail = data.user?.email;

  return isUserLogined ? (
    <div id="action" className="pr-4 flex self-center justify-end">
      <p className="inline-block mr-5">
        {userDisplayName ? userDisplayName : userEmail}
      </p>
      <a className="w-[24px] h-[24px] bg-center" href="/auth/logout">
        <LogoutIcon></LogoutIcon>
      </a>
    </div>
  ) : (
    <div id="action" className="pr-4 flex self-center justify-end">
      <div className="justify-center items-center flex-2 self-center w-20">
        <a href="/auth/login">LOGIN</a>
      </div>
      <div className="justify-center items-center flex-2 self-center border-spacing-1 text-center mr-5 rounded-md bg-purple-900 p-2">
        <a href="/auth/signup">START FOR FREE</a>
      </div>
    </div>
  );
};
