CD %cd%
@REM start "ngrok" ngrok start --config ./ngrok.yml --all
START "api" conda activate tf ^& uvicorn main:app --port 8001 --host 0.0.0.0 --reload