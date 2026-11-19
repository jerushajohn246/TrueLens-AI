@echo off
REM TruthLens - one-click start on Windows
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
if not exist samples\edited_photo.jpg python scripts\make_demo_samples.py
echo.
echo  TruthLens running at http://localhost:8000
echo.
python -m uvicorn app.main:app --reload --port 8000
