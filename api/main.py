from re import match

from fastapi import Depends, FastAPI, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stock.news import get_headernews
from stock.stock import Stock, StockCode

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

origins = [
    "http://localhost:3000",
    # "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stocks: list[Stock] = []
for stock_code in StockCode:
    stocks.append(Stock(stock_code.name))


@app.get("/news", response_class=JSONResponse)
def get_news():
    news = get_headernews(10)
    return news


@app.get("/stocks", response_class=JSONResponse)
def get_stocks():
    return stocks


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    news = get_headernews(10)
    context = {"request": request, "stocks": stocks, "title": "Home", "news": news}
    response = templates.TemplateResponse("stock/stocks.html", context)
    # response.set_cookie(key="session_key", value=session_key, expires=259200)  # 3 days
    return response


@app.get("/{name}/{date}", response_class=HTMLResponse)
def get_data(request: Request, name, date):
    current_stock = None
    current_prediction = None
    for stock in stocks:
        if stock.name == name:
            current_stock = stock
    if current_stock is not None:
        for prediction in current_stock.predictions:
            if prediction.date.strftime("%Y-%m-%d") == date:
                current_prediction = prediction
                break
    context = {
        "stock": current_stock,
        "prediction": current_prediction,
        "request": request,
        "name": name,
        "date": date,
    }
    response = templates.TemplateResponse("stock/stock.html", context)
    return response


@app.get("/back/{name}/{date}", response_class=HTMLResponse)
def back(request: Request, name, date):
    current_stock = None
    current_prediction = None
    for stock in stocks:
        if stock.name == name:
            current_stock = stock
    # match current_stock:
    #     case Stock:
    #         return
    if isinstance(current_stock, Stock):
        for prediction in current_stock.predictions:
            if prediction.date.strftime("%Y-%m-%d") == date:
                current_prediction = prediction
                break
    context = {
        "request": request,
        "stock": current_stock,
        "prediction": current_prediction,
    }
    response = templates.TemplateResponse("stock/stock-prediction.html", context)
    return response
