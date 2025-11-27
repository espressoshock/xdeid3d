#!/bin/bash
# X-DeID3D Experiments Viewer Launch Script
# Usage: ./launch_viewer.sh [experiments_dir]

set -e

echo "X-DeID3D Experiments Viewer"
echo "==========================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set experiments directory if provided
if [ -n "$1" ]; then
    export XDEID3D_EXPERIMENTS_DIR="$1"
    echo "Experiments directory: $XDEID3D_EXPERIMENTS_DIR"
fi

# Check dependencies
echo "Checking dependencies..."
pip show flask flask-cors > /dev/null 2>&1 || {
    echo "Installing dependencies..."
    pip install flask flask-cors --quiet
}

# Launch the viewer
echo ""
echo "Starting the viewer..."
echo "Access at: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the server"
echo "==========================="
echo ""

# Run the Flask app
cd backend
python app.py