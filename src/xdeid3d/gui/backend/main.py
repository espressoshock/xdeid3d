"""
X-DeID3D GUI Backend

FastAPI-based backend for the interactive 3D identity auditing GUI.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import generation, websocket, customization, camera

app = FastAPI(
    title="X-DeID3D GUI API",
    description="Backend API for X-DeID3D Identity Auditing Platform",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(generation.router, prefix="/api", tags=["generation"])
app.include_router(customization.router, prefix="/api/customization", tags=["customization"])
app.include_router(camera.router, prefix="/api/camera", tags=["camera"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

# Static files for outputs (mounted at /api/media for consistency)
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/api/media", StaticFiles(directory=str(OUTPUTS_DIR)), name="media")
app.mount("/api/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")  # Keep for backwards compatibility

# Static files for test pages
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/")
async def root():
    return {
        "name": "X-DeID3D GUI API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
