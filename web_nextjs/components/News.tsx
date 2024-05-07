interface INews {
  news: INew[];
}
const News = (props: INews) => {
  return (
    <div className="overflow-hidden">
      <div className="flex animate-tickerh hover:pause">
        {props.news.map((newObject: INew) => (
          <a
            key={newObject.url}
            className="underline text-[20px] flex-shrink-0 w-[50%] p-[10px]"
            target="_blank"
            href={newObject.url}
          >
            {newObject.title}
          </a>
        ))}
      </div>
    </div>
  );
};

export default News;
