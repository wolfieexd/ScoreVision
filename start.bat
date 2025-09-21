@echo off
REM ScoreVision Pro - Professional OMR Evaluation System
REM Windows Startup Script

echo.
echo 🚀 Starting ScoreVision Pro - Professional OMR Evaluation System
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Display Python version
echo 🐍 Python version:
python --version

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing dependencies...
if exist "requirements_final.txt" (
    pip install -r requirements_final.txt
) else (
    echo ❌ requirements_final.txt not found
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "app\static\uploads" mkdir "app\static\uploads"
if not exist "results" mkdir "results"
if not exist "answer_keys" mkdir "answer_keys"
if not exist "sample_data\sample_sheets" mkdir "sample_data\sample_sheets"
if not exist "production_results" mkdir "production_results"

REM Set environment variables
set FLASK_APP=app.py
set FLASK_ENV=development

echo.
echo ✅ Setup complete!
echo.
echo 🌐 Starting ScoreVision Pro on http://localhost:5000
echo 📊 Access the professional dashboard at http://localhost:5000
echo 🛠️  For API documentation, visit http://localhost:5000/api
echo.
echo 🔧 System Features:
echo    • Enterprise-grade OMR processing
echo    • Advanced computer vision algorithms
echo    • Real-time batch processing
echo    • Comprehensive analytics dashboard
echo    • Quality assurance ^& validation
echo.
echo Press Ctrl+C to stop the server
echo ==================================================
echo.

REM Start the Flask application
python app.py

pause