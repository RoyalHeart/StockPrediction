CD /D D:\Bon\VSCode\Project\StockPrediction\web
start "ngrok" ngrok start --config ../ngrok/ngrok.yml --all
conda activate tf && uvicorn main:app --port 8001 --host 0.0.0.0 --reload