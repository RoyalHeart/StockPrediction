from fastapi import Depends, FastAPI, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stock.stock import Stock

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
origins = ['http://localhost:3000','http://192.168.178.23:3000']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

stocks = [Stock("VIC"),Stock("ACB"),Stock("VJC")]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    context = {
        "request": request,
        "stocks": stocks,
        "title": "Home"
    }
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
            break
    context = {
        "stock": current_stock,
        "prediction": current_prediction,
        "request": request,
        "name" : name,
        "date" : date,
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
    context = {"request" : request,
                "stock": current_stock,
               "prediction": current_prediction}
    response = templates.TemplateResponse("stock/stock-prediction.html", context)
    return response

