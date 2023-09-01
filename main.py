import csv
from datetime import datetime

import pandas as pd
from openpyxl import load_workbook

from modules.cafef import get_cafef_dataset
from modules.fireant import get_fireant_dataset
from modules.model import ModelType, StockPrediction
from modules.ssi import get_ssi_dataset
from modules.stock import StockCode


def train_all_stock():
    for stock in StockCode:
        stock_model = StockPrediction(stock)
        stock_model.train(
            start_date=datetime(2021, 1, 1),
            model_type=ModelType.LSTM_DNN,
            epoch_size=200,
            batch_size=2,
            time_step=4,
            ratio=0.8,
        )


def predict_all_stock():
    predicted_prices = []
    change_percentages = []
    prices = []
    # end_date = "2023-06-09"
    end_date = datetime.today()
    for stock in StockCode:
        stock_model = StockPrediction(stock)
        predicted_price, change_percentage = stock_model.predict(
            predict_number=1, end_date=end_date
        )
        predicted_prices.append(predicted_price)
        change_percentages.append(float("%.4f" % change_percentage))
        price = get_ssi_dataset(
            stock.value, 1, start_date=datetime(2022, 1, 1), end_date=end_date
        )[0][0]
        prices.append(price)

    wb = load_workbook("./Stock.xlsx")
    df = pd.read_excel("./Stock.xlsx")
    # date_str = "09/06/2023"
    date_str = datetime.today().strftime("%d/%m/%Y")
    df[date_str] = prices
    df[f"{date_str} predict"] = predicted_prices
    df[f"{date_str} percent"] = change_percentages
    df.to_excel("./Stock.xlsx", index=False)
    # with open(
    #     f"./stock.csv",
    #     "w",
    #     encoding="UTF8",
    #     newline="",
    # ) as f:
    #     # create the csv writer
    #     writer = csv.reader(f)
    #     # write a row to the csv file
    #     writer.writerow([])
    #     print(f"write to file stock.csv")


def main():
    stock_model = StockPrediction(
        StockCode.VIC,
    )
    stock_model.train_data(
        start_date=datetime(2022, 1, 1),
        model_type=ModelType.LSTM_DNN,
        epoch_size=200,
        batch_size=4,
        time_step=4,
        ratio=0.8,
        ahead=0,
    )

    stock_model.predict(predict_number=1)
    stock_model = StockPrediction(
        StockCode.SSI,
    )
    stock_model.train_data(
        start_date=datetime(2022, 1, 1),
        model_type=ModelType.LSTM_DNN,
        epoch_size=200,
        batch_size=4,
        time_step=4,
        ratio=0.8,
        ahead=0,
    )
    stock_model.predict(predict_number=1)


def predict():
    # stock_model = StockPrediction(
    #     StockCode.BID,
    # )
    # stock_model.predict(predict_number=1)

    stock_model = StockPrediction(
        StockCode.VPB,
    )
    stock_model.predict(predict_number=1)
    # stock_model = StockPrediction(
    #     StockCode.VJC,
    # )
    # stock_model.predict(predict_number=1)


if __name__ == "__main__":
    # main()
    # predict()
    train_all_stock()
    # predict_all_stock()
