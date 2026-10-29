@echo off
REM One-click setup + launch for Windows. Double-click this file.
cd /d "%~dp0"
title RIA Co-pilot

where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python was not found. Install Python 3.11+ from https://www.python.org/downloads/
  echo and tick "Add Python to PATH" on the first installer screen, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv || (echo Failed to create venv & pause & exit /b 1)
)

if not exist ".venv\.installed" (
  echo Installing packages - first run only, takes 2-5 minutes...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Package install failed & pause & exit /b 1)
  echo done> ".venv\.installed"
)

if not exist ".env" copy ".env.example" ".env" >nul

if not exist "tests\fixtures\SYNTHETIC_Example_Liquid_Fund_SID.pdf" (
  ".venv\Scripts\python.exe" scripts\make_test_fixture.py
)

echo.
echo Running tests...
".venv\Scripts\python.exe" -m pytest -q
echo.
echo Starting the app at http://localhost:8501  (close this window or press Ctrl+C to stop)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
