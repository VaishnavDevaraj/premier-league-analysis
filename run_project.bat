@echo off
echo Premier League 2024-25 Analysis Project
echo =====================================
echo.

rem Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

rem Set project directory to where this batch file is located
set PROJECT_DIR=%~dp0
cd /d %PROJECT_DIR%

echo Setting up the project...
echo -------------------------

rem Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Setting up project structure...
python setup_project.py

echo.
echo Running data analysis demo...
python scripts\run_demo.py

echo.
echo Project setup and analysis complete!
echo.
echo Starting the interactive dashboard...
echo Dashboard will open in your web browser at http://localhost:8501
echo Press Ctrl+C in this window to stop the dashboard when finished.
echo.
streamlit run dashboard\app.py

rem Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat
