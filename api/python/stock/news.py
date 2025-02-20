import requests

VIETSTOCK_HEADER_NEWS_URL = "https://finance.vietstock.vn/data/headernews"
VIETSTOCK_STOCK_NEW_URL = "https://finance.vietstock.vn/data/getnewsbycode"
VIETSTOCK_BASE_URL = "https://vietstock.vn"


class New:
    def __init__(self, title, url):
        self.title = title
        self.url = url


def get_headernews(page_size=5):
    response = requests.post(
        url=VIETSTOCK_HEADER_NEWS_URL,
        data={"type": 1, "pageSize": page_size},
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36",
        },
    ).json()
    news = []
    for res in response:
        new = New(res["Title"], VIETSTOCK_BASE_URL + res["URL"])
        news.append(new)
    return news


def get_stock_news(stock_code: str, page_size=5) -> list[New]:
    response = requests.post(
        url=VIETSTOCK_STOCK_NEW_URL,
        data={
            "code": stock_code,
            "types[]": -1,
            "page": 1,
            "pageSize": page_size,
            "__RequestVerificationToken": "kkd1vsnepR9LjbiYDN7dzAsX0GIW90kfZk4F6ZaEeKfme4GOq6gxjzOWMO0qvbPivmQ9MsvrnAIpN7MbTkO2EQpVSS5sBrGrTsFwqRYGBMM1",
        },
        headers={
            "Accept": "*/*",
            "Content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36",
            "Cookie": "ASP.NET_SessionId=4otcznjr1hsrgwvlxadarubr; __RequestVerificationToken=7watNVWGGsFM78hAUSQFnLfWKDDiN6wgryJTI07K946sGME4gvH1DScEIRFaoiYJR2NnnNsvREWU87Yr6cskZjn3kLjQNuZW923K6FRkuTU1; language=vi-VN; Theme=Light; AnonymousNotification=; isShowLogin=true; finance_viewedstock=SAC,SSI,",
        },
    ).json()
    news = []
    for res in response[0]:
        new = New(res["Title"], VIETSTOCK_BASE_URL + res["URL"])
        news.append(new)
    return news
