#!/bin/bash

# X-DeID3D GUI - Stop Script

echo "🛑 Stopping X-DeID3D GUI servers..."

# Kill processes on ports
for port in 8000 5173; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo "   Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done

# Kill any lingering processes
pkill -f "vite" 2>/dev/null
pkill -f "npm run dev" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null

echo "✅ Servers stopped"
