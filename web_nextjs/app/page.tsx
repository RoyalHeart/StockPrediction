import News from "@/components/News";
import Stocks from "@/components/stock/Stocks";
import { createClient } from "@/utils/supabase/server";
import { redirect } from "next/navigation";
const backendApi = process.env.NEXT_PUBLIC_BACKEND_API;
const Home = async () => {
  const supabase = createClient();
  const { data, error } = await supabase.auth.getUser();
  if (error || !data?.user) {
    redirect("/login");
  }
  // const newsRes = await fetch(`${backendApi}/news`);
  // const news: INew[] = await newsRes.json();
  // const stocksRes = await fetch(`${backendApi}/stocks`, {
  //   headers: { "Content-Type": "application/json" },
  // });
  // const stocks: IStock[] = await stocksRes.json();

  const main = (
    <>
      <div className="flex sticky top-0 bg-black/30 backdrop-blur-sm">
        <div className="flex-1">
          <a href="/">
            <h1 className="text-2xl/loose pl-[3%] bg-gradient-to-r from-green-400 from-25%  to-purple-500 to-80% inline-block text-transparent bg-clip-text">
              Trad<span className="">i</span>ng AI
            </h1>
          </a>
        </div>
        <nav className="justify-center items-center flex-1 self-center">
          <a href="/about">ABOUT</a>
        </nav>
        <div className="justify-center items-center flex-1 self-center">
          <a href="/news">NEWS</a>
        </div>
        <div className="justify-center items-center flex-2 self-center w-20">
          <a href="/login">LOGIN</a>
        </div>
        <div className="justify-center items-center flex-2 self-center w-20 border-spacing-1 text-center mr-5 rounded-md bg-purple-900 p-1">
          <a href="/signin">SIGN UP</a>
        </div>
        <div>{data.user.email}</div>
      </div>
      {/* <News news={news} /> */}
      {/* <Stocks stocks={stocks} /> */}
      <div id="footer">
        <h5>
          Source: news are from <a href="https://vietstock.vn">vietstock.vn</a>{" "}
          and stock data is from{" "}
          <a href="https://iboard.ssi.com.vn/">iboard.ssi.com.vn</a> and the
          website is made by{" "}
          <a href="https://github.com/RoyalHeart">RoyalHeart</a>
        </h5>
      </div>
    </>
  );
  return main;
};

export default Home;
