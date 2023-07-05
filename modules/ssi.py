import datetime

import numpy as np
import requests

SSI_URL = "https://iboard.ssi.com.vn/dchart/api/history"
# SSI_URL = "https://iboard.ssi.com.vn/dchart/api/history?resolution=D&symbol=ACB&from=1652498660&to=1686626720"


def create_params(
    stock_code: str,
    start_date: datetime.datetime = datetime.datetime(2022, 1, 1, 0, 0),
    end_date: datetime.datetime = datetime.datetime.today(),
):
    params = {
        "resolution": "D",
        "symbol": stock_code,
        "from": str(start_date.timestamp().__floor__()),
        "to": str(end_date.timestamp().__floor__()),
    }
    return params


def get_ssi_dataset(
    stock_code: str,
    limit: int,
    start_date: datetime.datetime = datetime.datetime(2022, 1, 1, 0, 0),
    end_date: datetime.datetime = (datetime.datetime.today()),
):
    params = create_params(stock_code, start_date, end_date)
    response = requests.get(
        SSI_URL,
        params=params,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36",
        },
    ).json()
    close_prices_res = response["c"]
    close_prices = []
    for i in range(len(close_prices_res)):
        close_prices.append(close_prices_res[i])
    dataset = np.asarray(close_prices, "float32")
    dataset = np.reshape(dataset, (-1, 1))
    return dataset[-limit:]


def get_ssi_dataset_two(
    stock_code: str,
    limit: int,
    start_date: datetime.datetime = datetime.datetime(2022, 1, 1, 0, 0),
    end_date: datetime.datetime = (datetime.datetime.today()),
):
    params = create_params(stock_code, start_date, end_date)
    response = requests.get(
        SSI_URL,
        params=params,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36",
        },
    ).json()
    close_prices_res = response["c"]
    volumes_res = response["v"]
    close_prices = []
    for i in range(len(close_prices_res)):
        close_prices.append([close_prices_res[i], volumes_res[i]])
    dataset = np.asarray(close_prices, "float32")
    dataset = np.reshape(dataset, (-1, 2))
    return dataset[-limit:]


# print(get_ssi_dataset_two("ACB", 5))
