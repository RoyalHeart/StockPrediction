import datetime

import numpy as np
import requests

CAFEF_URL = "https://msh-data.cafef.vn/graphql/"


def create_graphql_body(
    stock_code: str,
    start_date: datetime.date = datetime.date(year=2022, month=1, day=1),
    end_date: datetime.date = datetime.date.today(),
):
    query = (
        'query{\r\n  tradingViewData(symbol: "'
        + stock_code
        + '", from: "'
        + start_date.strftime("%Y-%m-%d")
        + '",to: "'
        + end_date.strftime("%Y-%m-%d")
        + '") {\r\n  symbol\r\n  open\r\n  close\r\n  high\r\n  low\r\n  volume\r\n  time\r\n }}'
    )
    # query = (
    #     "query{\r\n  tradingViewData(symbol: "
    #     + stock_code
    #     + ", from: "
    #     + start_date
    #     + ",to: "
    #     + end_date
    #     + ") {\r\n  symbol\r\n  open\r\n  close\r\n  high\r\n  low\r\n  volume\r\n  time\r\n }}"
    # )

    return {"query": query}


def get_cafef_dataset(
    stock_code: str,
    limit: int,
    start_date: datetime.date = datetime.date(year=2022, month=1, day=1),
    end_date: datetime.date = (datetime.date.today()),
):
    start_date
    body = create_graphql_body(stock_code, start_date, end_date)
    response = requests.post(CAFEF_URL, json=body).json()
    data_list = response["data"]["tradingViewData"]
    priceCloses = []
    for daily in data_list:
        priceCloses.append(daily["close"])
    dataset = np.asarray(priceCloses, "float32")
    dataset = np.reshape(dataset, (-1, 1))
    return dataset[-limit:]


# print(get_cafef_dataset("ACB", 5))
