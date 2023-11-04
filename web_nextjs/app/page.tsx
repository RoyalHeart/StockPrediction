import News from "@/components/News";
import Stocks from "@/components/stock/Stocks";
const Home = async () => {
  const newsRes = await fetch(`http://localhost:8001/news`);
  const news: INew[] = await newsRes.json();
  const stocksRes = await fetch("http://localhost:8001/stocks", {
    headers: { "Content-Type": "application/json" },
  });
  const stocks: IStock[] = await stocksRes.json();

  const main = (
    <>
      <h1 className="text-4xl ml-[3%]">Hello, have a nice day!</h1>
      <News news={news} />
      <Stocks stocks={stocks} />
      <div id="footer">
        <h5>
          Source: news are from <a href="https://vietstock.vn">vietstock.vn</a>{" "}
          and stock data is from{" "}
          <a href="https://iboard.ssi.com.vn/">iboard.ssi.com.vn</a>
          and the website is made by{" "}
          <a href="https://github.com/RoyalHeart">RoyalHeart</a>
        </h5>
      </div>
    </>
  );
  return main;
};

export default Home;
