@echo off
REM One-click run on Windows. Needs Python only -- the compiled UI is served
REM by the backend, so Node.js is not required to run or demo the app.
setlocal
cd /d "%~dp0.."

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found on your PATH.
  echo Install it from https://www.python.org/downloads/ and tick
  echo "Add python.exe to PATH" during setup, then run this again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv || goto :fail
)

echo Installing dependencies...
.venv\Scripts\python -m pip install --quiet --upgrade pip
.venv\Scripts\python -m pip install --quiet -r requirements.txt || goto :fail

if not exist ".env" (
  echo Creating .env from .env.example...
  copy /y .env.example .env >nul
)

echo.
echo Starting webhook receiver on port 9000...
start "camera-agent receiver" cmd /c "cd backend && ..\.venv\Scripts\python -m uvicorn app.receiver:app --host 127.0.0.1 --port 9000"

echo Starting app on port 8000...
echo.
echo   Open http://localhost:8000 in your browser.
echo   Press Ctrl+C in this window to stop.
echo.
cd backend
..\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
goto :eof

:fail
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
