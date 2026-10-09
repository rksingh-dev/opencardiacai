@echo off
echo ========================================================
echo   Starting OpenCardiac AI REST API Server (Port 8000)
echo ========================================================
echo.
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
pause
