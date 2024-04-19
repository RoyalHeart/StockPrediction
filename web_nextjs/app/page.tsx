import News from "@/components/News";
import Stocks from "@/components/stock/Stocks";
const Home = async () => {
  // const newsRes = await fetch(`http://localhost:8001/news`);
  // const news: INew[] = await newsRes.json();
  // const stocksRes = await fetch("http://localhost:8001/stocks", {
  //   headers: { "Content-Type": "application/json" },
  // });
  // const stocks: IStock[] = await stocksRes.json();

  const main = (
    <>
      <div className="flex sticky top-0 bg-black/30 backdrop-blur-sm">
        <div className="flex-1">
          <a href="/">
            <h1 className="text-4xl/loose pl-[3%] pt-[1%] bg-gradient-to-r from-green-400 from-25%  to-purple-500 to-80% inline-block text-transparent bg-clip-text">
              Trad<span className="">i</span>ng AI
            </h1>
          </a>
        </div>
        <nav className="justify-center items-center flex-1">
          <a href="/stocks">STOCKS</a>
        </nav>
        <div className="flex-1">
          <a href="/stocks">STOCKS</a>
        </div>
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
