import News from "@/components/News";
import Stocks from "@/components/stock/Stocks";
import { createSupabaseServerClient } from "@/utils/supabase/server";
import { redirect } from "next/navigation";
const backendApi = process.env.NEXT_PUBLIC_BACKEND_API;
import LogoutIcon from "@mui/icons-material/Logout";
import { NavBar } from "@/components/navbar/NavBar";
const Home = async () => {
  // const newsRes = await fetch(`${backendApi}/news`);
  // const news: INew[] = await newsRes.json();
  // const stocksRes = await fetch(`${backendApi}/stocks`, {
  //   headers: { "Content-Type": "application/json" },
  // });
  // const stocks: IStock[] = await stocksRes.json();
  const main = (
    <>
      <div className="flex flex-col h-screen justify-between">
        <NavBar></NavBar>
        {/* <News news={news} /> */}
        {/* <Stocks stocks={stocks} /> */}
        <div id="content" className="mb-auto"></div>
        <footer id="footer" className="self-center">
          <h5 className="">
            @ 2024, Source: news are from{" "}
            <a href="https://vietstock.vn">vietstock.vn</a> and stock data is
            from <a href="https://iboard.ssi.com.vn/">iboard.ssi.com.vn</a> and
            the website is made by{" "}
            <a href="https://github.com/RoyalHeart">RoyalHeart</a>
          </h5>
        </footer>
      </div>
    </>
  );
  return main;
};

export default Home;
