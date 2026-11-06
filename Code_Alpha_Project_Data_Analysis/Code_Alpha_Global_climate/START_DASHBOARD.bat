@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Please run setup first.
  pause
  exit /b 1
)

echo Starting AirLens dashboard...
echo Open http://127.0.0.1:8501 if a browser does not open automatically.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address=127.0.0.1

pause
