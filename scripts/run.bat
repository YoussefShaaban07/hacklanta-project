@echo off
REM One-click run on Windows. Needs Python only -- the compiled UI is served
REM by the backend, so Node.js is not required to run or demo the app.
setlocal
cd /d "%~dp0.."

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python was not found on your PATH.
  echo Install it from https://www.python.org/downloads/ and tick
  echo "Add python.exe to PATH" during setup, then run this again.
  echo.
  pause
  exit /b 1
)

REM Check the interpreter version BEFORE a long install. Too old is a hard
REM stop. Very new is only a warning: wheels usually appear within a few
REM months, and blocking a version that works would be wrong. If wheels are
REM genuinely missing, pip falls back to a Rust source build and the
REM :depfail message below explains what to do.
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo.
  echo ---------------------------------------------------------------
  echo  Python is too old:
  python --version
  echo  This project needs Python 3.10 or newer.
  echo  Get 3.12 from https://www.python.org/downloads/
  echo ---------------------------------------------------------------
  echo.
  pause
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info < (3,14) else 1)"
if errorlevel 1 (
  echo.
  echo  Note: you are on a very new Python:
  python --version
  echo  Some dependencies may not ship wheels for it yet. Trying anyway;
  echo  if the install fails, see the advice printed at the end.
  echo.
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv || goto :fail
)

echo Installing dependencies ^(this takes a minute the first time^)...
.venv\Scripts\python -m pip install --quiet --upgrade pip
REM --prefer-binary keeps pip on wheels instead of source builds.
.venv\Scripts\python -m pip install --prefer-binary -r requirements.txt
if errorlevel 1 goto :depfail

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

:depfail
echo.
echo ---------------------------------------------------------------
echo  Dependency installation failed.
echo.
echo  If the output above mentions "cargo", "maturin" or
echo  "failed building wheel", pip could not find a prebuilt wheel
echo  for your Python version and tried to compile from source.
echo.
echo  Your Python:
python --version
echo.
echo  Fix: install Python 3.12, delete the .venv folder, re-run this.
echo    https://www.python.org/downloads/release/python-3128/
echo ---------------------------------------------------------------
echo.
pause
exit /b 1

:fail
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
