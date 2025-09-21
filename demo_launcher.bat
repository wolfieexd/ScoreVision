@echo off
REM ScoreVision Pro - Complete Demo Launcher
REM Starts both Flask Backend and Streamlit Frontend

echo.
echo 🎯 ScoreVision Pro - Complete Demo Launcher
echo ============================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

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

echo.
echo ✅ Setup complete!
echo.
echo 🚀 Choose Your Interface:
echo.
echo [1] Streamlit MVP Interface (Recommended for Evaluators)
echo     📊 Modern dashboard with real-time analytics
echo     🎯 Perfect for daily operations and demonstrations
echo     🌐 Access at: http://localhost:8501
echo.
echo [2] Complete Flask Web Application
echo     🏢 Full enterprise interface
echo     📈 Advanced administration and reporting
echo     🌐 Access at: http://localhost:5000
echo.
echo [3] Launch Both Interfaces (Advanced)
echo     🔥 Run both frontend and backend simultaneously
echo     🌐 Streamlit: http://localhost:8501
echo     🌐 Flask: http://localhost:5000
echo.
set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" goto streamlit
if "%choice%"=="2" goto flask
if "%choice%"=="3" goto both
echo ❌ Invalid choice. Please select 1, 2, or 3.
pause
exit /b 1

:streamlit
echo.
echo 🎯 Starting Streamlit MVP Interface...
echo 📊 Perfect for evaluators and daily operations
echo 🌐 Access the dashboard at: http://localhost:8501
echo.
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
goto end

:flask
echo.
echo 🏢 Starting Flask Web Application...
echo 📈 Complete enterprise interface
echo 🌐 Access the application at: http://localhost:5000
echo.
python app.py
goto end

:both
echo.
echo 🔥 Starting Both Interfaces...
echo.
echo 🎯 Streamlit MVP will be available at: http://localhost:8501
echo 🏢 Flask Application will be available at: http://localhost:5000
echo.
echo Starting Flask backend in background...
start /b python app.py
timeout /t 3 /nobreak >nul
echo Starting Streamlit frontend...
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

:end
pause