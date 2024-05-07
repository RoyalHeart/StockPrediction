import { createSupabaseServerClient } from "@/utils/supabase/server";

const AccountPage = async () => {
  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase.auth.getUser();
  if (error || !data?.user) {
    // redirect("/");
  }
  return <>{data.user?.email}</>;
};

export default AccountPage;
