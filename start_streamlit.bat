@echo off
REM ScoreVision Pro - Streamlit MVP Frontend
REM Windows Startup Script for Evaluator Interface

echo.
echo 🎯 Starting ScoreVision Pro - Streamlit MVP Interface
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

REM Install requirements including Streamlit
echo 📥 Installing dependencies (including Streamlit)...
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

echo.
echo ✅ Setup complete!
echo.
echo 🌐 Starting ScoreVision Pro Streamlit MVP Interface
echo 📊 Access the evaluator dashboard at the URL shown below
echo 🔧 This interface is optimized for evaluators and administrators
echo.
echo 💡 Features Available:
echo    • Professional Dashboard with Real-time Analytics
echo    • Upload ^& Process OMR Sheets
echo    • Batch Processing with Progress Tracking
echo    • Results Review ^& Quality Control
echo    • System Configuration ^& Settings
echo.
echo 🚀 Backend Integration:
echo    • Automatically connects to Flask backend (if running)
echo    • Standalone mode available for demonstrations
echo    • Real-time system status monitoring
echo.
echo Press Ctrl+C to stop the interface
echo ==================================================
echo.

REM Start Streamlit application
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

pause