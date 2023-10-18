from fastapi import Depends, FastAPI, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stock.news import get_headernews
from stock.stock import Stock

from modules.stock import StockCode

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


stocks = []
for stock_code in StockCode:
    stocks.append(Stock(stock_code.name))
# stocks = [Stock("SSI"), Stock("POW"), Stock("VIC"), Stock("ACB"), Stock("VJC")]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    news = get_headernews(10)
    print(stocks[0].news)
    context = {"request": request, "stocks": stocks, "title": "Home", "news": news}
    response = templates.TemplateResponse("index.html", context)
    # response.set_cookie(key="session_key", value=session_key, expires=259200)  # 3 days
    return response


@app.get("/{name}/{date}", response_class=HTMLResponse)
def get_data(request: Request, name, date):
    current_stock = None
    current_prediction = None
    for stock in stocks:
        if stock.name == name:
            current_stock = stock
    for prediction in current_stock.predictions:
        if prediction.date.strftime("%Y-%m-%d") == date:
            current_prediction = prediction
            print(current_prediction.predicted_price)
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
    for prediction in current_stock.predictions:
        if prediction.date.strftime("%Y-%m-%d") == date:
            current_prediction = prediction
            break
    print(current_prediction, current_stock)
    context = {
        "request": request,
        "stock": current_stock,
        "prediction": current_prediction,
    }
    response = templates.TemplateResponse("stock/stock-prediction.html", context)
    return response
