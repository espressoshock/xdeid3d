#!/bin/bash
# Launch X-DeID3D Interactive GUI
# Usage: ./launch_gui.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "Starting X-DeID3D Interactive GUI..."

# Check if we should run in development or production mode
if [ "$1" = "--dev" ]; then
    echo "Running in development mode..."

    # Start backend in background
    echo "Starting backend on port 8000..."
    cd "$BACKEND_DIR"
    python main.py &
    BACKEND_PID=$!

    # Start frontend dev server
    echo "Starting frontend dev server on port 5173..."
    cd "$FRONTEND_DIR"
    npm run dev

    # Cleanup on exit
    kill $BACKEND_PID 2>/dev/null
else
    echo "Running backend only (use --dev for frontend development)..."
    cd "$BACKEND_DIR"
    python main.py
fi
