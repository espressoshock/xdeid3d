"""
FastAPI application for X-DeID3D interactive GUI.

Provides a web-based interface for:
- Running evaluations on images and videos
- Generating heatmaps and visualizations
- Comparing anonymization methods
- Managing experiments
"""

import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["create_app", "GUIConfig", "run_gui"]


@dataclass
class GUIConfig:
    """Configuration for the GUI application.

    Attributes:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
        upload_dir: Directory for uploaded files
        output_dir: Directory for generated outputs
        max_upload_size: Maximum upload size in bytes
    """
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    upload_dir: Optional[str] = None
    output_dir: Optional[str] = None
    max_upload_size: int = 100 * 1024 * 1024  # 100MB

    def __post_init__(self):
        if self.upload_dir is None:
            self.upload_dir = os.path.join(tempfile.gettempdir(), "xdeid3d_uploads")
        if self.output_dir is None:
            self.output_dir = os.path.join(tempfile.gettempdir(), "xdeid3d_outputs")

        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# Task storage for async operations
_tasks: Dict[str, Dict[str, Any]] = {}


def create_app(config: GUIConfig = None):
    """Create and configure the FastAPI application.

    Args:
        config: GUI configuration

    Returns:
        Configured FastAPI application
    """
    try:
        from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError(
            "FastAPI is required for the GUI. "
            "Install with: pip install fastapi uvicorn python-multipart"
        )

    if config is None:
        config = GUIConfig()

    app = FastAPI(
        title="X-DeID3D GUI",
        description="Interactive evaluation and visualization for face anonymization",
        version="0.1.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store config
    app.state.config = config

    # Mount static files
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app):
    """Register all API routes."""
    from fastapi import File, UploadFile, HTTPException, BackgroundTasks, Form
    from fastapi.responses import FileResponse, JSONResponse

    @app.get("/")
    async def index():
        """Serve the main page."""
        frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "X-DeID3D GUI API", "docs": "/docs"}

    @app.get("/api/status")
    async def get_status():
        """Get API status."""
        return {
            "status": "running",
            "version": "0.1.0",
        }

    @app.get("/api/anonymizers")
    async def list_anonymizers():
        """List available anonymizers."""
        try:
            from xdeid3d.anonymizers import list_anonymizers
            return {"anonymizers": list_anonymizers()}
        except Exception as e:
            return {"anonymizers": {}, "error": str(e)}

    @app.get("/api/metrics")
    async def list_metrics():
        """List available metrics."""
        try:
            from xdeid3d.metrics import list_metrics
            return {"metrics": list_metrics()}
        except Exception as e:
            return {"metrics": {}, "error": str(e)}

    @app.post("/api/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload a file."""
        config = app.state.config

        # Generate unique filename
        ext = os.path.splitext(file.filename)[1]
        new_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(config.upload_dir, new_filename)

        # Save file
        content = await file.read()
        if len(content) > config.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {config.max_upload_size // (1024*1024)}MB"
            )

        with open(file_path, "wb") as f:
            f.write(content)

        return {
            "filename": new_filename,
            "original_name": file.filename,
            "size": len(content),
            "path": file_path,
        }

    @app.post("/api/evaluate/single")
    async def evaluate_single(
        background_tasks: BackgroundTasks,
        original: UploadFile = File(...),
        anonymized: UploadFile = File(...),
        metrics: str = Form("standard"),
    ):
        """Evaluate a single image pair."""
        config = app.state.config

        # Save uploaded files
        orig_path = os.path.join(config.upload_dir, f"orig_{uuid.uuid4().hex}.png")
        anon_path = os.path.join(config.upload_dir, f"anon_{uuid.uuid4().hex}.png")

        with open(orig_path, "wb") as f:
            f.write(await original.read())
        with open(anon_path, "wb") as f:
            f.write(await anonymized.read())

        # Create task
        task_id = uuid.uuid4().hex
        _tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "result": None,
            "error": None,
        }

        # Run evaluation in background
        background_tasks.add_task(
            _run_evaluation,
            task_id,
            orig_path,
            anon_path,
            metrics,
        )

        return {"task_id": task_id, "status": "started"}

    @app.get("/api/tasks/{task_id}")
    async def get_task_status(task_id: str):
        """Get task status."""
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        return _tasks[task_id]

    @app.post("/api/generate/heatmap")
    async def generate_heatmap(
        background_tasks: BackgroundTasks,
        data_file: UploadFile = File(...),
        colormap: str = Form("magma"),
        resolution: int = Form(72),
    ):
        """Generate heatmap from evaluation data."""
        config = app.state.config

        # Save uploaded file
        data_path = os.path.join(config.upload_dir, f"data_{uuid.uuid4().hex}.json")
        with open(data_path, "wb") as f:
            f.write(await data_file.read())

        # Create task
        task_id = uuid.uuid4().hex
        _tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "result": None,
            "error": None,
        }

        # Run in background
        background_tasks.add_task(
            _generate_heatmap,
            task_id,
            data_path,
            config.output_dir,
            colormap,
            resolution,
        )

        return {"task_id": task_id, "status": "started"}

    @app.get("/api/outputs/{filename}")
    async def get_output(filename: str):
        """Get generated output file."""
        config = app.state.config
        file_path = os.path.join(config.output_dir, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(file_path)


async def _run_evaluation(
    task_id: str,
    original_path: str,
    anonymized_path: str,
    metrics_str: str,
):
    """Background task for running evaluation."""
    try:
        import cv2
        from xdeid3d.metrics import get_metric
        from xdeid3d.cli.utils import parse_metrics

        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 10

        # Load images
        original = cv2.imread(original_path)
        anonymized = cv2.imread(anonymized_path)

        if original is None or anonymized is None:
            raise ValueError("Failed to load images")

        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        anonymized = cv2.cvtColor(anonymized, cv2.COLOR_BGR2RGB)

        _tasks[task_id]["progress"] = 30

        # Parse metrics
        metric_names = parse_metrics(metrics_str)

        # Compute metrics
        results = {}
        for i, name in enumerate(metric_names):
            try:
                metric = get_metric(name)
                score = metric.compute(original, anonymized)
                results[name] = float(score)
            except Exception as e:
                results[name] = {"error": str(e)}

            progress = 30 + int(60 * (i + 1) / len(metric_names))
            _tasks[task_id]["progress"] = progress

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["progress"] = 100
        _tasks[task_id]["result"] = results

        # Cleanup
        os.remove(original_path)
        os.remove(anonymized_path)

    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


async def _generate_heatmap(
    task_id: str,
    data_path: str,
    output_dir: str,
    colormap: str,
    resolution: int,
):
    """Background task for generating heatmap."""
    try:
        import json
        from xdeid3d.visualization import HeatmapGenerator, save_figure

        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 10

        # Load data
        with open(data_path) as f:
            data = json.load(f)

        _tasks[task_id]["progress"] = 30

        # Extract frame metrics
        if "frame_metrics" in data:
            frame_data = data["frame_metrics"]
        elif isinstance(data, list):
            frame_data = data
        else:
            raise ValueError("Unknown data format")

        # Create heatmap generator
        generator = HeatmapGenerator(resolution=resolution)

        # Find metric to visualize
        sample = frame_data[0]
        metric_name = None
        for key in sample.keys():
            if key not in ("yaw", "pitch", "frame_idx", "frame_index", "timestamp"):
                if isinstance(sample[key], (int, float)):
                    metric_name = key
                    break

        if metric_name is None:
            raise ValueError("No numeric metric found")

        generator.set_metric_name(metric_name)

        _tasks[task_id]["progress"] = 50

        # Add scores
        for d in frame_data:
            yaw = d.get("yaw", 0)
            pitch = d.get("pitch", np.pi / 2)
            score = d.get(metric_name, 0)
            generator.add_score(float(yaw), float(pitch), float(score))

        _tasks[task_id]["progress"] = 70

        # Generate heatmap
        heatmap = generator.generate()
        rgb = heatmap.to_rgb(colormap=colormap)

        _tasks[task_id]["progress"] = 90

        # Save output
        output_filename = f"heatmap_{task_id}.png"
        output_path = os.path.join(output_dir, output_filename)

        from PIL import Image
        img = Image.fromarray(rgb)
        img.save(output_path)

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["progress"] = 100
        _tasks[task_id]["result"] = {
            "filename": output_filename,
            "metric": metric_name,
            "resolution": resolution,
        }

        # Cleanup
        os.remove(data_path)

    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


def run_gui(
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
) -> None:
    """Run the GUI application.

    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode

    Example:
        >>> run_gui(port=8000)
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the GUI. "
            "Install with: pip install uvicorn"
        )

    config = GUIConfig(host=host, port=port, debug=debug)
    app = create_app(config)

    print(f"Starting X-DeID3D GUI")
    print(f"Access the application at: http://localhost:{port}")
    print(f"API documentation at: http://localhost:{port}/docs")

    uvicorn.run(app, host=host, port=port, reload=debug)


if __name__ == "__main__":
    run_gui()
