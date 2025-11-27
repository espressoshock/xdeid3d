#!/bin/bash

# X-DeID3D GUI - Restart Script
# Stops and restarts both backend and frontend

echo "🔄 X-DeID3D GUI Restart Script"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to kill process on port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}Killing process on port $port (PID: $pid)${NC}"
        kill -9 $pid 2>/dev/null
        sleep 1
    else
        echo -e "${GREEN}Port $port is free${NC}"
    fi
}

# Stop existing processes
echo ""
echo "📛 Stopping existing processes..."
kill_port 8000  # Backend
kill_port 5173  # Frontend

# Kill any lingering npm/vite processes
pkill -f "vite" 2>/dev/null
pkill -f "npm run dev" 2>/dev/null

echo ""
echo "✅ Processes stopped"
echo ""

# Check if we should start servers
read -p "Start servers now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Exiting without starting servers."
    exit 0
fi

# Start backend in background
echo ""
echo "🚀 Starting backend..."
cd backend
source activate SphereHead 2>/dev/null || conda activate SphereHead 2>/dev/null || true
python main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"
echo "   Logs: gui/backend.log"
echo "   URL: http://localhost:8000"

# Wait for backend to start
echo "   Waiting for backend to initialize..."
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Backend health check passed${NC}"
else
    echo -e "${RED}   ❌ Backend health check failed${NC}"
    echo "   Check backend.log for errors"
fi

# Start frontend in background
echo ""
echo "🚀 Starting frontend..."
cd ../frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"
echo "   Logs: gui/frontend.log"
echo "   URL: http://localhost:5173"

echo ""
echo "========================================"
echo -e "${GREEN}✅ Servers started!${NC}"
echo ""
echo "Backend:  http://localhost:8000  (PID: $BACKEND_PID)"
echo "Frontend: http://localhost:5173  (PID: $FRONTEND_PID)"
echo ""
echo "Logs:"
echo "  Backend:  tail -f gui/backend.log"
echo "  Frontend: tail -f gui/frontend.log"
echo ""
echo "To stop:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo "  or run: ./stop.sh"
echo ""
echo "Open in browser: http://localhost:5173"
echo "========================================"
