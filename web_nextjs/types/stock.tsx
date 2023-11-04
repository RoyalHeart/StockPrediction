interface IPrediction {
  date: string;
  price: number;
  predicted_price: number;
  predicted_ratio: number;
  result: string;
}

interface IStock {
  predictions: IPrediction[];
  name: string;
  news: INew[];
}
