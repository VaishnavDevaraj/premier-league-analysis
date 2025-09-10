@echo off
echo Premier League 2024-25 Analysis Dashboard
echo =========================================
echo.

rem Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

rem Check Python version
python --version | findstr /r "3\.[89]\|3\.1[0-9]" >nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Python version may not be compatible
    echo Recommended: Python 3.8 or higher
    echo.
    echo Press any key to continue anyway or Ctrl+C to exit...
    pause >nul
)

echo Setting up environment...

rem Check if venv exists, if not create it
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

rem Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

rem Check if requirements are installed
if not exist venv\Lib\site-packages\streamlit (
    echo Installing required packages...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
)

echo.
echo Starting the Premier League Dashboard...
echo.
echo NOTE: The dashboard will open in your default web browser.
echo Press Ctrl+C in this window to stop the dashboard when finished.
echo.
echo Loading...

rem Run the dashboard
streamlit run dashboard/app.py

rem Deactivate virtual environment when done
call venv\Scripts\deactivate

pause
