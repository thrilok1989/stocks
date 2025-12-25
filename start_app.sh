#!/bin/bash

echo "========================================="
echo "🚀 NIFTY/SENSEX Trading Dashboard"
echo "   Python Backend + HTML Frontend"
echo "========================================="
echo ""

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask dependencies..."
    pip install -r requirements_flask.txt
    echo "✅ Dependencies installed"
    echo ""
fi

echo "🔧 Starting Flask Backend..."
echo "📍 Backend: http://localhost:5000"
echo "🌐 Frontend: http://localhost:5000/"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

# Start Flask backend (which serves the frontend)
python3 flask_backend.py
