import StockDisplay from "./StockDisplay";
import StockNew from "./StockNew";

interface IStocks {
  stocks: IStock[];
}
const Stocks = (props: IStocks) => {
  return (
    <div className="flex flex-col flex-wrap">
      {props.stocks.map((stock) => (
        <div
          key={stock.name}
          className="m-[10px] relative inline-block self-center w-[90%] bg-black/30
         rounded-md z-[2]"
        >
          <div className="inline-block relative z-[-1] w-[10%] left-[10px] p-[5px] opacity-100 text-lg">
            {stock.name}
          </div>
          <StockNew news={stock.news} />
          <StockDisplay predictions={stock.predictions} name={stock.name} />
        </div>
      ))}
    </div>
  );
};

export default Stocks;
