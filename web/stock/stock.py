import random
import sys

sys.path.insert(0, "..")
from modules.ssi import get_ssi_dataset_date

from .news import get_stock_news


class Prediction:
    def __init__(self, date, price, predicted_price):
        self.date = date
        self.price = price
        self.predicted_price = predicted_price
        self.predicted_ratio = round((predicted_price - self.price) / self.price, 4)
        self.result = "true" if random.random() > 0.5 else "false"


class Stock:
    def __init__(self, name: str):
        self.name = name
        self.predictions = self.get_data()
        self.news = self.get_news()

    def get_data(self) -> list[Prediction]:
        datas, dates = get_ssi_dataset_date(self.name, 10)
        predictions = []
        for i in range(10):
            date = dates[i].date()
            price = datas[i][0]
            random_predicted_price = round(price + random.random() * 10 - 5, 2)
            prediction = Prediction(date, price, random_predicted_price)
            predictions.append(prediction)
        return predictions

    def get_news(self):
        return get_stock_news(self.name, 5)
