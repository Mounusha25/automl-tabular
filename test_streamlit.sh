#!/bin/bash
# Test Streamlit app locally before deploying

echo "🧪 Testing Streamlit App Locally..."
echo ""
echo "This will open http://localhost:8501 in your browser"
echo "Press Ctrl+C to stop the server"
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not installed. Installing..."
    pip3 install streamlit --quiet
fi

# Run the app
python3 -m streamlit run app/streamlit_app.py
