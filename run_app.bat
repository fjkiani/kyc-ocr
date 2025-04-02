@echo off
REM Document Processing System Launcher Script for Windows

echo ======================================================
echo        Document Processing System Launcher
echo ======================================================

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% LSS 3 (
    echo Python 3.8 or higher is required.
    echo Current version: %PYTHON_VERSION%
    pause
    exit /b 1
)

if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 8 (
        echo Python 3.8 or higher is required.
        echo Current version: %PYTHON_VERSION%
        pause
        exit /b 1
    )
)

echo Python %PYTHON_VERSION% detected.

REM Check for system dependencies
echo Checking system dependencies...

REM Windows users need to install dependencies manually
echo IMPORTANT: For PDF processing and file type detection, you need:
echo 1. Poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases/
echo 2. libmagic for Windows: See https://github.com/ahupp/python-magic#dependencies
echo.
echo After downloading, add the bin directories to your PATH environment variable.
echo.
set /p CONTINUE=Have you installed these dependencies? (y/n): 
if /i "%CONTINUE%" NEQ "y" (
    echo.
    echo You can continue without these dependencies, but PDF processing
    echo and file type detection may not work correctly.
    echo.
    set /p CONTINUE=Continue anyway? (y/n): 
    if /i "%CONTINUE%" NEQ "y" (
        pause
        exit /b 1
    )
)

REM Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Virtual environment activated.

REM Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo Checking dependencies...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install dependencies.
        set /p CONTINUE=Continue anyway? (y/n): 
        if /i "%CONTINUE%" NEQ "y" (
            pause
            exit /b 1
        )
    ) else (
        echo Dependencies installed.
    )
)

REM Create temp directory if it doesn't exist
if not exist temp (
    echo Creating temp directory...
    mkdir temp
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create temp directory.
        set /p CONTINUE=Continue anyway? (y/n): 
        if /i "%CONTINUE%" NEQ "y" (
            pause
            exit /b 1
        )
    ) else (
        echo Temp directory created.
    )
)

REM Check if test directory has sample images
set HAS_IMAGES=0
if exist test\*.jpg set HAS_IMAGES=1
if exist test\*.jpeg set HAS_IMAGES=1
if exist test\*.png set HAS_IMAGES=1

if %HAS_IMAGES% EQU 0 (
    echo No sample images found. Downloading sample images...
    if exist download_sample_images.py (
        python download_sample_images.py
        if %ERRORLEVEL% NEQ 0 (
            echo Failed to download sample images.
            set /p CONTINUE=Continue anyway? (y/n): 
            if /i "%CONTINUE%" NEQ "y" (
                pause
                exit /b 1
            )
        ) else (
            echo Sample images downloaded successfully.
        )
    ) else (
        echo Sample image downloader script not found.
        set /p CONTINUE=Continue anyway? (y/n): 
        if /i "%CONTINUE%" NEQ "y" (
            pause
            exit /b 1
        )
    )
)

REM Check if .env file exists
if not exist .env (
    echo No .env file found. Creating one...
    set /p API_KEY=Please enter your Fireworks API key: 
    echo FIREWORKS_API_KEY=%API_KEY%> .env
    echo .env file created.
)

REM Run the application
echo ======================================================
echo        Starting Document Processing System
echo ======================================================

REM Run with Python if run_app.py exists, otherwise use streamlit directly
if exist run_app.py (
    python run_app.py
) else if exist app.py (
    streamlit run app.py
) else (
    echo No app.py or run_app.py found.
    pause
    exit /b 1
)

pause 