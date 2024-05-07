import { User } from "@supabase/supabase-js";
import LogoutIcon from "@mui/icons-material/Logout";
import { createSupabaseServerClient } from "@/utils/supabase/server";
export const NavActions = async () => {
  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase.auth.getUser();
  let isUserLogined: boolean = true;
  if (error || !data?.user) {
    isUserLogined = false;
  }
  return isUserLogined ? (
    <div id="action" className="pr-4 flex self-center justify-end">
      <p className="inline-block mr-5">{data.user?.email}</p>
      <a className="w-[24px] h-[24px] bg-center" href="/auth/logout">
        <LogoutIcon></LogoutIcon>
      </a>
    </div>
  ) : (
    <div id="action" className="pr-4 flex self-center justify-end">
      <div className="justify-center items-center flex-2 self-center w-20">
        <a href="/auth/login">LOGIN</a>
      </div>
      <div className="justify-center items-center flex-2 self-center w-20 border-spacing-1 text-center mr-5 rounded-md bg-purple-900 p-1">
        <a href="/auth/signup">SIGN UP</a>
      </div>
    </div>
  );
};
