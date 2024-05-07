const StockNew = (props: INews) => {
  return (
    <div className="overflow-hidden h-[30px] inline-block text-center w-[80%]">
      <div className="relative p-[5px] flex flex-col animate-tickerv hover:pause">
        {props.news.map((newObj) => (
          <a
            key={newObj.url}
            className="underline text-center leading-[30px] text-lg flex-shrink-0 w-[100%]"
            target="_blank"
            href={newObj.url}
          >
            {newObj.title}
          </a>
        ))}
      </div>
    </div>
  );
};
export default StockNew;
