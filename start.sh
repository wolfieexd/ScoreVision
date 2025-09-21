#!/bin/bash

# ScoreVision Pro - Professional OMR Evaluation System
# Startup Script for Development/Production

echo "🚀 Starting ScoreVision Pro - Professional OMR Evaluation System"
echo "=================================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
if [ -f "requirements_final.txt" ]; then
    pip install -r requirements_final.txt
else
    echo "❌ requirements_final.txt not found"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p app/static/uploads
mkdir -p results
mkdir -p answer_keys
mkdir -p sample_data/sample_sheets
mkdir -p production_results

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting ScoreVision Pro on http://localhost:5000"
echo "📊 Access the professional dashboard at http://localhost:5000"
echo "🛠️  For API documentation, visit http://localhost:5000/api"
echo ""
echo "🔧 System Features:"
echo "   • Enterprise-grade OMR processing"
echo "   • Advanced computer vision algorithms"
echo "   • Real-time batch processing"
echo "   • Comprehensive analytics dashboard"
echo "   • Quality assurance & validation"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================="

# Start the Flask application
python app.py