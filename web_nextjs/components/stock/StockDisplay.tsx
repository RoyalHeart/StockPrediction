interface IStockDisplay {
  predictions: IPrediction[];
  name: string;
}
const StockDisplay = (props: IStockDisplay) => {
  return (
    <div className="flex flex-row justify-start flex-wrap text-center">
      {props.predictions.map((prediction) => (
        <div
          key={prediction.date}
          hx-get={"http://localhost:8001/" + props.name + "/" + prediction.date}
          hx-swap="outerHTML transition:true"
          hx-boost="false"
          className="m-[10px] bg-[#000064]/30 rounded-2xl hover:pointer-events-auto hover:bg-[#000064]/10"
        >
          <div className="m-[10px]">{prediction.date}</div>
          <div className="m-[10px]">{prediction.price}</div>
          {prediction.predicted_ratio > 0 ? (
            <div className="text-c_green m-[10px]">UP</div>
          ) : prediction.predicted_ratio == 0 ? (
            <div className="text-c_orange m-[10px]">UNCHANGED</div>
          ) : (
            <div className="text-c_red m-[10px]">DOWN</div>
          )}
          {prediction.result ? (
            <div className="text-c_green m-[10px]">✓</div>
          ) : (
            <div className="text-c_red">✕</div>
          )}
        </div>
      ))}
    </div>
  );
};

export default StockDisplay;
