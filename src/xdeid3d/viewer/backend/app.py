"""
Flask backend for X-DeID3D Experiments Viewer

This is the web dashboard for viewing and analyzing evaluation experiments.
"""

import os
import sys
from flask import Flask, jsonify, send_file, send_from_directory, request, Response, make_response
from flask_cors import CORS
from pathlib import Path
import mimetypes
import subprocess
import tempfile
import hashlib

from utils.experiment_reader import ExperimentReader

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Initialize experiment reader
# Default to 'experiments' directory relative to the viewer package
VIEWER_DIR = Path(__file__).parent.parent
EXPERIMENTS_DIR = os.environ.get(
    'XDEID3D_EXPERIMENTS_DIR',
    str(VIEWER_DIR.parent.parent.parent / 'experiments')
)
reader = ExperimentReader(EXPERIMENTS_DIR)

# Create cache directory for transcoded videos
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.video_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


@app.route('/')
def index():
    """Serve the main frontend page"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/experiments')
def get_experiments():
    """Get all experiments"""
    experiments = reader.get_all_experiments()
    return jsonify({
        'experiments': experiments,
        'total': len(experiments)
    })


@app.route('/api/experiments/<cfg>/<int:seed>')
def get_experiment(cfg: str, seed: int):
    """Get specific experiment details"""
    experiment = reader.get_experiment_info(cfg, seed)
    if experiment:
        return jsonify(experiment)
    return jsonify({'error': 'Experiment not found'}), 404


@app.route('/api/test/file/<path:filepath>')
def test_file(filepath):
    """Test endpoint to check if file exists and is readable"""
    file_path = os.path.join(EXPERIMENTS_DIR, filepath)
    
    result = {
        'requested_path': filepath,
        'full_path': file_path,
        'exists': os.path.exists(file_path),
        'is_file': os.path.isfile(file_path) if os.path.exists(file_path) else False,
        'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        'readable': os.access(file_path, os.R_OK) if os.path.exists(file_path) else False
    }
    
    return jsonify(result)


@app.route('/api/metrics/summary')
def get_metrics_summary():
    """Get metrics summary across all experiments"""
    summary = reader.get_metrics_summary()
    return jsonify(summary)


@app.route('/api/comparisons')
def get_comparisons():
    """Get experiments grouped for comparison"""
    cfg = request.args.get('cfg', None)
    comparisons = reader.get_experiment_comparisons(cfg)
    return jsonify(comparisons)


def check_ffmpeg_available():
    """Check if ffmpeg and ffprobe are available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except:
        return False


def get_video_codec(video_path):
    """Check video codec using ffprobe"""
    if not check_ffmpeg_available():
        return None
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name', '-of', 'default=nw=1:nk=1',
            video_path
        ], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return None


def transcode_video_if_needed(video_path, cache_dir):
    """Transcode video to H.264 if needed"""
    # If ffmpeg not available, return original
    if not check_ffmpeg_available():
        print("FFmpeg not available, serving original video")
        return video_path
        
    codec = get_video_codec(video_path)
    
    # If we can't detect codec or it's already H.264, return original
    if not codec or codec == 'h264':
        return video_path
    
    # Generate cache filename based on original path
    path_hash = hashlib.md5(video_path.encode()).hexdigest()
    cache_filename = f"{path_hash}_h264.mp4"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # If already transcoded, return cached version
    if os.path.exists(cache_path):
        print(f"Serving cached H.264 version: {cache_path}")
        return cache_path
    
    # Transcode to H.264
    try:
        print(f"Transcoding {video_path} from {codec} to h264...")
        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y',  # Overwrite output file
            cache_path
        ], check=True, capture_output=True)
        print(f"Transcoded successfully to {cache_path}")
        return cache_path
    except subprocess.CalledProcessError as e:
        print(f"Transcoding failed: {e}")
        # Try simpler transcoding without audio
        try:
            print("Trying transcoding without audio...")
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-an',  # No audio
                '-movflags', '+faststart',
                '-y',
                cache_path
            ], check=True, capture_output=True)
            print(f"Transcoded successfully (no audio) to {cache_path}")
            return cache_path
        except:
            print("All transcoding attempts failed, serving original")
            return video_path


@app.route('/api/media/<path:filepath>')
def serve_media(filepath):
    """Serve media files (videos, images, meshes)"""
    file_path = os.path.join(EXPERIMENTS_DIR, filepath)
    
    if os.path.exists(file_path):
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return jsonify({'error': 'File is empty'}), 400
            
        # Determine mime type based on extension
        ext = os.path.splitext(filepath)[1].lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.ply': 'application/octet-stream',
            '.json': 'application/json'
        }
        
        mime_type = mime_types.get(ext, 'application/octet-stream')
        
        # For video files, check codec and transcode if needed
        if ext == '.mp4':
            try:
                # Check if video needs transcoding
                video_to_serve = transcode_video_if_needed(file_path, CACHE_DIR)
                
                return send_file(
                    video_to_serve,
                    mimetype=mime_type,
                    as_attachment=False,
                    conditional=True,
                    max_age=0
                )
                    
            except Exception as e:
                print(f"Error serving video {filepath}: {str(e)}")
                return jsonify({'error': f'Error serving video file: {str(e)}'}), 500
        else:
            return send_file(file_path, mimetype=mime_type)
    
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/download/<path:filepath>')
def download_file(filepath):
    """Download experiment files"""
    file_path = os.path.join(EXPERIMENTS_DIR, filepath)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    
    return jsonify({'error': 'File not found'}), 404


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    # For non-API routes, serve the frontend (for client-side routing)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    # Check if experiments directory exists
    if not os.path.exists(EXPERIMENTS_DIR):
        print(f"Warning: Experiments directory not found at {EXPERIMENTS_DIR}")
        print("Creating empty experiments directory...")
        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    print(f"Starting X-DeID3D Experiments Viewer")
    print(f"Experiments directory: {EXPERIMENTS_DIR}")
    print(f"Access the viewer at: http://localhost:5001")

    app.run(debug=True, host='0.0.0.0', port=5001)