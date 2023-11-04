CD %cd%
@REM start "ngrok" ngrok start --config ./ngrok.yml --all
start "htmx" conda activate tf ^&^& uvicorn main:app --port 8002 --host 0.0.0.0 --reload