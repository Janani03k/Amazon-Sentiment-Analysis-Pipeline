@echo off
echo Killing processes on port 8001...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8001') do (
    echo Killing PID %%a
    taskkill /F /PID %%a
)
echo All processes using port 8001 killed.
echo Starting Uvicorn on port 8001...
uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
pause
