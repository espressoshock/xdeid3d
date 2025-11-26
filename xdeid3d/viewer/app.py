"""
Flask application for X-DeID3D Experiments Viewer.

Provides a web interface for browsing, comparing, and visualizing
evaluation experiments and their results.
"""

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, send_file, send_from_directory, request

__all__ = ["create_app", "ViewerConfig", "run_viewer"]


@dataclass
class ViewerConfig:
    """Configuration for the experiments viewer.

    Attributes:
        experiments_dir: Directory containing experiment results
        cache_dir: Directory for video transcoding cache
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    experiments_dir: str = "./experiments"
    cache_dir: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 5001
    debug: bool = False

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = os.path.join(
                os.path.dirname(__file__),
                ".video_cache"
            )


def create_app(config: ViewerConfig = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config: Viewer configuration

    Returns:
        Configured Flask application
    """
    if config is None:
        config = ViewerConfig()

    # Determine static folder path
    static_folder = os.path.join(os.path.dirname(__file__), "frontend")

    app = Flask(__name__, static_folder=static_folder, static_url_path="")

    # Enable CORS
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        pass  # CORS optional

    # Store config
    app.config["VIEWER_CONFIG"] = config
    app.config["EXPERIMENTS_DIR"] = os.path.abspath(config.experiments_dir)

    # Create cache directory
    os.makedirs(config.cache_dir, exist_ok=True)

    # Initialize experiment reader
    from xdeid3d.viewer.experiment_reader import ExperimentReader
    app.config["READER"] = ExperimentReader(app.config["EXPERIMENTS_DIR"])

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: Flask) -> None:
    """Register all API routes."""

    @app.route("/")
    def index():
        """Serve the main frontend page."""
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/experiments")
    def get_experiments():
        """Get all experiments."""
        reader = app.config["READER"]
        experiments = reader.get_all_experiments()
        return jsonify({
            "experiments": experiments,
            "total": len(experiments)
        })

    @app.route("/api/experiments/<cfg>/<int:seed>")
    def get_experiment(cfg: str, seed: int):
        """Get specific experiment details."""
        reader = app.config["READER"]
        experiment = reader.get_experiment_info(cfg, seed)
        if experiment:
            return jsonify(experiment)
        return jsonify({"error": "Experiment not found"}), 404

    @app.route("/api/metrics/summary")
    def get_metrics_summary():
        """Get metrics summary across all experiments."""
        reader = app.config["READER"]
        summary = reader.get_metrics_summary()
        return jsonify(summary)

    @app.route("/api/comparisons")
    def get_comparisons():
        """Get experiments grouped for comparison."""
        reader = app.config["READER"]
        cfg = request.args.get("cfg", None)
        comparisons = reader.get_experiment_comparisons(cfg)
        return jsonify(comparisons)

    @app.route("/api/media/<path:filepath>")
    def serve_media(filepath):
        """Serve media files (videos, images, meshes)."""
        experiments_dir = app.config["EXPERIMENTS_DIR"]
        config = app.config["VIEWER_CONFIG"]

        file_path = os.path.join(experiments_dir, filepath)

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return jsonify({"error": "File is empty"}), 400

        # Determine mime type
        ext = os.path.splitext(filepath)[1].lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".ply": "application/octet-stream",
            ".json": "application/json",
            ".npz": "application/octet-stream",
        }
        mime_type = mime_types.get(ext, "application/octet-stream")

        # For video files, check codec and transcode if needed
        if ext == ".mp4":
            try:
                video_to_serve = _transcode_video_if_needed(
                    file_path,
                    config.cache_dir
                )
                return send_file(
                    video_to_serve,
                    mimetype=mime_type,
                    as_attachment=False,
                    conditional=True,
                    max_age=0
                )
            except Exception as e:
                return jsonify({"error": f"Error serving video: {e}"}), 500

        return send_file(file_path, mimetype=mime_type)

    @app.route("/api/download/<path:filepath>")
    def download_file(filepath):
        """Download experiment files."""
        experiments_dir = app.config["EXPERIMENTS_DIR"]
        file_path = os.path.join(experiments_dir, filepath)

        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)

        return jsonify({"error": "File not found"}), 404

    @app.route("/api/test/file/<path:filepath>")
    def test_file(filepath):
        """Test endpoint to check if file exists."""
        experiments_dir = app.config["EXPERIMENTS_DIR"]
        file_path = os.path.join(experiments_dir, filepath)

        return jsonify({
            "requested_path": filepath,
            "full_path": file_path,
            "exists": os.path.exists(file_path),
            "is_file": os.path.isfile(file_path) if os.path.exists(file_path) else False,
            "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        })

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        if request.path.startswith("/api/"):
            return jsonify({"error": "API endpoint not found"}), 404
        # For non-API routes, serve the frontend
        return send_from_directory(app.static_folder, "index.html")


def _check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _get_video_codec(video_path: str) -> Optional[str]:
    """Get video codec using ffprobe."""
    if not _check_ffmpeg_available():
        return None

    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1",
            video_path
        ], capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def _transcode_video_if_needed(video_path: str, cache_dir: str) -> str:
    """Transcode video to H.264 if needed for browser compatibility."""
    if not _check_ffmpeg_available():
        return video_path

    codec = _get_video_codec(video_path)

    # If already H.264, return original
    if not codec or codec == "h264":
        return video_path

    # Generate cache filename
    path_hash = hashlib.md5(video_path.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{path_hash}_h264.mp4")

    # Return cached version if exists
    if os.path.exists(cache_path):
        return cache_path

    # Transcode to H.264
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y",
            cache_path
        ], check=True, capture_output=True)
        return cache_path

    except subprocess.CalledProcessError:
        # Try without audio
        try:
            subprocess.run([
                "ffmpeg", "-i", video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-an",
                "-movflags", "+faststart",
                "-y",
                cache_path
            ], check=True, capture_output=True)
            return cache_path
        except Exception:
            return video_path


def run_viewer(
    experiments_dir: str = "./experiments",
    host: str = "0.0.0.0",
    port: int = 5001,
    debug: bool = False,
) -> None:
    """Run the experiments viewer.

    Args:
        experiments_dir: Directory containing experiments
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode

    Example:
        >>> run_viewer("./experiments", port=5001)
    """
    config = ViewerConfig(
        experiments_dir=experiments_dir,
        host=host,
        port=port,
        debug=debug,
    )

    app = create_app(config)

    print(f"Starting X-DeID3D Experiments Viewer")
    print(f"Experiments directory: {config.experiments_dir}")
    print(f"Access the viewer at: http://localhost:{port}")

    app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    run_viewer()
