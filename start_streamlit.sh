#!/bin/bash
# ScoreVision Pro - Streamlit MVP Frontend
# Unix/Linux/macOS Startup Script for Evaluator Interface

echo ""
echo "🎯 Starting ScoreVision Pro - Streamlit MVP Interface"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Display Python version
echo "🐍 Python version:"
python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements including Streamlit
echo "📥 Installing dependencies (including Streamlit)..."
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

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting ScoreVision Pro Streamlit MVP Interface"
echo "📊 Access the evaluator dashboard at the URL shown below"
echo "🔧 This interface is optimized for evaluators and administrators"
echo ""
echo "💡 Features Available:"
echo "   • Professional Dashboard with Real-time Analytics"
echo "   • Upload & Process OMR Sheets"
echo "   • Batch Processing with Progress Tracking"
echo "   • Results Review & Quality Control"
echo "   • System Configuration & Settings"
echo ""
echo "🚀 Backend Integration:"
echo "   • Automatically connects to Flask backend (if running)"
echo "   • Standalone mode available for demonstrations"
echo "   • Real-time system status monitoring"
echo ""
echo "Press Ctrl+C to stop the interface"
echo "=================================================="
echo ""

# Start Streamlit application
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0