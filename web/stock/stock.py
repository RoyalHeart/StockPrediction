import random
import sys

sys.path.insert(0, '..')
from modules.ssi import get_ssi_dataset_date


class Prediction:
    def __init__(self,  date, price, predicted_price):
        self.date = date
        self.price = price
        self.predicted_price = predicted_price
        self.predicted_ratio = (self.predicted_price - self.price) / self.price
        self.result = 'true' if random.random() > 0.5 else 'false'

class Stock:
    def __init__(self, name: str):
        self.name = name
        self.predictions = self.get_data()
    def get_data(self) -> list[Prediction]:
        datas, dates = get_ssi_dataset_date(self.name, 20)
        predictions = []
        for i in range(20):
            prediction = Prediction(dates[i].date(), datas[i][0], random.random() * 10 + datas[i][0]-5)
            predictions.append(prediction)
        return predictions

