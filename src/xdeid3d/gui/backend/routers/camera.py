"""
Camera Pose Customization API routes
Handles 3D mesh generation and custom camera trajectory video synthesis
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import os
import sys

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.mesh_service import get_mesh_service
from services.camera_pose_service import get_camera_pose_service
from routers.websocket import send_to_session

router = APIRouter()

# Configuration
MODEL_PATH = os.getenv('SPHEREHEAD_MODEL_PATH', str(Path(__file__).parent.parent.parent.parent / 'models' / 'spherehead-ckpt-025000.pkl'))
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
CORE_DIR = Path(__file__).parent.parent.parent / "core"

def url_to_filesystem_path(url_path: Optional[str]) -> Optional[str]:
    """
    Convert URL path to filesystem path.

    Handles paths like:
    - /api/outputs/session/... -> /full/path/to/outputs/session/...
    - /api/media/session/... -> /full/path/to/outputs/session/...

    Args:
        url_path: URL path (e.g., /api/outputs/session/file.npz)

    Returns:
        Absolute filesystem path or None if input is None
    """
    if not url_path:
        return None

    # Remove /api/outputs or /api/media prefix
    if url_path.startswith('/api/outputs/'):
        relative_path = url_path.replace('/api/outputs/', '')
    elif url_path.startswith('/api/media/'):
        relative_path = url_path.replace('/api/media/', '')
    else:
        # Already a filesystem path or unknown format
        print(f"[url_to_filesystem_path] Path doesn't start with /api/outputs or /api/media: {url_path}")
        return url_path

    # Construct absolute filesystem path
    filesystem_path = str(OUTPUTS_DIR / relative_path)
    print(f"[url_to_filesystem_path] Converted {url_path} -> {filesystem_path}")
    return filesystem_path


# === Request Models ===

class CameraKeyframe(BaseModel):
    """Camera keyframe with pose parameters"""
    timestamp: float  # 0.0 to 1.0 (normalized time)
    yaw: float        # radians
    pitch: float      # radians
    radius: float     # camera distance


class MeshGenerationRequest(BaseModel):
    """Request to generate 3D mesh from identity"""
    session_id: str
    latent_code_path: Optional[str] = None  # For PTI-based identities
    generator_path: Optional[str] = None    # For PTI fine-tuned generator
    seed: Optional[int] = None              # For seed-based identities
    truncation: float = 0.65
    voxel_res: int = 512  # Mesh resolution (512 or 1024)
    force_regenerate: bool = False  # Force regeneration even if cached mesh exists


class PresetVideoRequest(BaseModel):
    """Request to generate video with preset camera trajectory"""
    session_id: str
    preset_id: str  # e.g., "rotate360", "orbit", "sidebyside"
    duration_seconds: float = 10.0
    num_frames: int = 240
    # Identity parameters
    latent_code_path: Optional[str] = None
    generator_path: Optional[str] = None
    seed: Optional[int] = None
    truncation: float = 0.65
    # Background settings (to re-apply from stage 2.1)
    apply_background: bool = False
    background_color: Optional[str] = None
    background_image_path: Optional[str] = None


class KeyframeVideoRequest(BaseModel):
    """Request to generate video with custom keyframe trajectory"""
    session_id: str
    keyframes: List[CameraKeyframe]
    interpolation: str = "cubic"  # "linear" or "cubic"
    num_frames: int = 240
    # Identity parameters
    latent_code_path: Optional[str] = None
    generator_path: Optional[str] = None
    seed: Optional[int] = None
    truncation: float = 0.65
    # Background settings
    apply_background: bool = False
    background_color: Optional[str] = None
    background_image_path: Optional[str] = None


class PreviewRequest(BaseModel):
    """Request to render preview with camera pose"""
    session_id: str
    mode: str  # 'preset' or 'keyframes'
    # For preset mode
    preset_id: Optional[str] = None
    # For keyframe mode
    keyframes: Optional[List[CameraKeyframe]] = None
    interpolation: str = "cubic"
    # Preview settings
    duration_seconds: Optional[float] = None  # Duration from timeline (NEW)
    preview_frames: Optional[int] = None  # Fallback if duration not provided
    preview_fps: int = 24     # 24 or 30
    preview_quality: str = 'medium'  # 'low', 'medium', 'high'
    # Identity parameters
    latent_code_path: Optional[str] = None
    generator_path: Optional[str] = None
    seed: Optional[int] = None
    truncation: float = 0.65


# === Response Models ===

class MeshGenerationResponse(BaseModel):
    """Response from mesh generation request"""
    success: bool
    session_id: str
    mesh_url: Optional[str] = None
    task_id: Optional[str] = None
    message: str
    error: Optional[str] = None


class VideoGenerationResponse(BaseModel):
    """Response from video generation request"""
    success: bool
    session_id: str
    video_url: Optional[str] = None
    task_id: Optional[str] = None
    message: str
    error: Optional[str] = None


class PreviewResponse(BaseModel):
    """Response from preview rendering request"""
    success: bool
    session_id: str
    preview_url: Optional[str] = None
    preview_type: str = 'sequence'  # 'frame' or 'sequence'
    message: str
    error: Optional[str] = None


# === API Endpoints ===

@router.post("/mesh", response_model=MeshGenerationResponse)
async def generate_mesh(request: MeshGenerationRequest):
    """
    Generate 3D mesh from identity using shape extraction

    This will:
    1. Load the latent code (from PTI or seed)
    2. Run gen_videos.py with --shapes flag
    3. Extract .ply mesh file
    4. Return mesh URL for 3D viewer
    """
    try:
        # Convert URL paths to filesystem paths
        latent_code_path = url_to_filesystem_path(request.latent_code_path)
        generator_path = url_to_filesystem_path(request.generator_path)

        print(f"[Camera/Mesh] Request - session: {request.session_id}, latent: {latent_code_path}, seed: {request.seed}")

        # Validate that we have either latent_code_path or seed
        if not latent_code_path and request.seed is None:
            raise HTTPException(
                status_code=400,
                detail="Must provide either latent_code_path (for PTI) or seed (for seed-based generation)"
            )

        # Validate latent code path if provided
        if latent_code_path and not os.path.exists(latent_code_path):
            print(f"[Camera/Mesh] ERROR: Latent code file not found at: {latent_code_path}")
            raise HTTPException(status_code=404, detail=f"Latent code not found: {latent_code_path}")

        # Create output directory
        output_dir = OUTPUTS_DIR / request.session_id / "mesh"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get mesh service
        mesh_service = get_mesh_service(MODEL_PATH)

        # Define progress callback to send updates via WebSocket
        async def progress_callback(progress: float, message: str):
            """Send mesh generation progress to frontend via WebSocket"""
            await send_to_session(request.session_id, {
                "type": "mesh_progress",
                "progress": progress,
                "message": message,
            })

        # Generate mesh (this will take several minutes)
        print(f"[Camera/Mesh] Starting mesh generation (force_regenerate={request.force_regenerate})...")
        mesh_path = await mesh_service.generate_mesh(
            session_id=request.session_id,
            output_dir=output_dir,
            latent_code_path=latent_code_path,
            generator_path=generator_path,
            seed=request.seed,
            truncation=request.truncation,
            voxel_res=request.voxel_res,
            force_regenerate=request.force_regenerate,
            progress_callback=progress_callback,
        )

        print(f"[Camera/Mesh] Mesh generated successfully: {mesh_path}")

        # Generate URL for frontend
        # mesh_path is like: /path/to/outputs/session_id/mesh/seed0000.ply
        # We want: /api/media/session_id/mesh/seed0000.ply
        mesh_filename = Path(mesh_path).name
        mesh_url = f"/api/media/{request.session_id}/mesh/{mesh_filename}"

        return MeshGenerationResponse(
            success=True,
            session_id=request.session_id,
            mesh_url=mesh_url,
            message="Mesh generated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return MeshGenerationResponse(
            success=False,
            session_id=request.session_id,
            message="Failed to generate mesh",
            error=str(e),
        )


@router.post("/preview", response_model=PreviewResponse)
async def render_preview(request: PreviewRequest):
    """
    Render preview video with custom camera pose

    This generates a quick preview video optimized for fast rendering:
    - Configurable quality (low/medium/high)
    - Configurable frame count (30-240)
    - Configurable FPS (24, 30)
    - Works with both preset and keyframe modes
    """
    try:
        # Convert URL paths to filesystem paths
        latent_code_path = url_to_filesystem_path(request.latent_code_path)
        generator_path = url_to_filesystem_path(request.generator_path)

        print(f"[Camera/Preview] Request - session: {request.session_id}, mode: {request.mode}")

        # Calculate number of frames from duration (NEW)
        if request.duration_seconds:
            # Use timeline duration to calculate frames
            num_frames = int(request.duration_seconds * request.preview_fps)
            print(f"[Camera/Preview] Using duration: {request.duration_seconds}s @ {request.preview_fps}fps = {num_frames} frames")
        elif request.preview_frames:
            # Fallback to explicit frame count
            num_frames = request.preview_frames
            print(f"[Camera/Preview] Using explicit frame count: {num_frames} frames")
        else:
            # Default fallback
            num_frames = 60
            print(f"[Camera/Preview] Using default frame count: {num_frames} frames")

        print(f"[Camera/Preview] Settings - frames: {num_frames}, fps: {request.preview_fps}, quality: {request.preview_quality}")

        # Validate mode
        if request.mode not in ['preset', 'keyframes']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{request.mode}'. Must be 'preset' or 'keyframes'"
            )

        # Validate preset or keyframes based on mode
        if request.mode == 'preset':
            if not request.preset_id:
                raise HTTPException(status_code=400, detail="preset_id required for preset mode")
            valid_presets = ['rotate360', 'orbit', 'sidebyside', 'front']
            if request.preset_id not in valid_presets:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid preset '{request.preset_id}'. Must be one of: {valid_presets}"
                )
        elif request.mode == 'keyframes':
            if not request.keyframes or len(request.keyframes) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="At least 2 keyframes required for keyframes mode"
                )

        # Validate identity parameters
        if not latent_code_path and request.seed is None:
            raise HTTPException(status_code=400, detail="Must provide either latent_code_path or seed")

        if latent_code_path and not os.path.exists(latent_code_path):
            raise HTTPException(status_code=404, detail=f"Latent code not found: {latent_code_path}")

        # Create output directory
        output_dir = OUTPUTS_DIR / request.session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get camera pose service
        pose_service = get_camera_pose_service(MODEL_PATH)

        # Generate trajectory based on mode
        if request.mode == 'preset':
            trajectory = pose_service.get_preset_trajectory(
                preset_id=request.preset_id,
                num_frames=num_frames,
                duration=num_frames / request.preview_fps,  # Calculate duration from frames and fps
            )
            print(f"[Camera/Preview] Generated preset trajectory '{request.preset_id}' with {len(trajectory)} frames")
        else:  # keyframes
            keyframes_dict = [
                {
                    'timestamp': kf.timestamp,
                    'yaw': kf.yaw,
                    'pitch': kf.pitch,
                    'radius': kf.radius,
                }
                for kf in request.keyframes
            ]
            trajectory = pose_service.interpolate_keyframes(
                keyframes=keyframes_dict,
                num_frames=num_frames,
                interpolation=request.interpolation,
            )
            print(f"[Camera/Preview] Interpolated trajectory from {len(request.keyframes)} keyframes to {len(trajectory)} frames")

        # Define progress callback to send updates via WebSocket
        async def progress_callback(progress: float, message: str):
            """Send preview rendering progress to frontend via WebSocket"""
            await send_to_session(request.session_id, {
                "type": "pose_preview_progress",
                "progress": progress,
                "message": message,
            })

        # Render preview
        print(f"[Camera/Preview] Starting preview rendering...")
        preview_path = await pose_service.render_preview(
            session_id=request.session_id,
            output_dir=output_dir,
            trajectory=trajectory,
            latent_code_path=latent_code_path,
            generator_path=generator_path,
            seed=request.seed,
            truncation=request.truncation,
            num_frames=num_frames,
            fps=request.preview_fps,
            quality=request.preview_quality,
            progress_callback=progress_callback,
        )

        print(f"[Camera/Preview] Preview rendered successfully: {preview_path}")

        # Generate URL for frontend with cache-busting timestamp
        # preview_path is like: /path/to/outputs/session_id/previews/preview_session_id.mp4
        # We want: /api/media/session_id/previews/preview_session_id.mp4?t=timestamp
        import time
        preview_filename = Path(preview_path).name
        timestamp = int(time.time() * 1000)  # Milliseconds since epoch
        preview_url = f"/api/media/{request.session_id}/previews/{preview_filename}?t={timestamp}"
        print(f"[Camera/Preview] Generated preview URL with cache-busting: {preview_url}")

        return PreviewResponse(
            success=True,
            session_id=request.session_id,
            preview_url=preview_url,
            preview_type='sequence',
            message=f"Preview rendered successfully ({num_frames} frames @ {request.preview_fps}fps, {request.preview_quality} quality)",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return PreviewResponse(
            success=False,
            session_id=request.session_id,
            message="Failed to render preview",
            error=str(e),
        )


@router.post("/preset", response_model=VideoGenerationResponse)
async def generate_preset_video(request: PresetVideoRequest):
    """
    Generate video with preset camera trajectory

    Available presets:
    - rotate360: Full 360° horizontal rotation
    - orbit: Orbital path with pitch variation
    - sidebyside: Left-right oscillation
    - front: Static frontal view
    """
    try:
        # Convert URL paths to filesystem paths
        latent_code_path = url_to_filesystem_path(request.latent_code_path)
        generator_path = url_to_filesystem_path(request.generator_path)

        print(f"[Camera/Preset] Request - session: {request.session_id}, preset: {request.preset_id}")

        # Validate preset
        valid_presets = ['rotate360', 'orbit', 'sidebyside', 'front']
        if request.preset_id not in valid_presets:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid preset '{request.preset_id}'. Must be one of: {valid_presets}"
            )

        # Validate identity parameters
        if not latent_code_path and request.seed is None:
            raise HTTPException(status_code=400, detail="Must provide either latent_code_path or seed")

        if latent_code_path and not os.path.exists(latent_code_path):
            raise HTTPException(status_code=404, detail=f"Latent code not found: {latent_code_path}")

        # Create output directory
        output_dir = OUTPUTS_DIR / request.session_id / "camera_pose"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get camera pose service
        pose_service = get_camera_pose_service(MODEL_PATH)

        # Generate trajectory from preset
        trajectory = pose_service.get_preset_trajectory(
            preset_id=request.preset_id,
            num_frames=request.num_frames,
            duration=request.duration_seconds,
        )

        print(f"[Camera/Preset] Generated trajectory with {len(trajectory)} frames")

        # Generate video
        video_path = await pose_service.generate_pose_video(
            session_id=request.session_id,
            output_dir=output_dir,
            trajectory=trajectory,
            latent_code_path=latent_code_path,
            generator_path=generator_path,
            seed=request.seed,
            truncation=request.truncation,
            num_frames=request.num_frames,
            progress_callback=None,  # TODO: Add WebSocket progress
        )

        print(f"[Camera/Preset] Video generated: {video_path}")

        # Generate URL (for now, returning trajectory file as placeholder)
        video_filename = Path(video_path).name
        video_url = f"/api/media/{request.session_id}/camera_pose/{video_filename}"

        return VideoGenerationResponse(
            success=True,
            session_id=request.session_id,
            video_url=video_url,
            message=f"Preset video generated (placeholder - actual video generation not yet implemented)",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return VideoGenerationResponse(
            success=False,
            session_id=request.session_id,
            message="Failed to generate preset video",
            error=str(e),
        )


@router.post("/keyframes", response_model=VideoGenerationResponse)
async def generate_keyframe_video(request: KeyframeVideoRequest):
    """
    Generate video with custom keyframe trajectory

    This will:
    1. Interpolate camera poses between keyframes
    2. Generate video frames at each pose
    3. Optionally re-apply background customization
    4. Return video URL
    """
    try:
        # Convert URL paths to filesystem paths
        latent_code_path = url_to_filesystem_path(request.latent_code_path)
        generator_path = url_to_filesystem_path(request.generator_path)

        print(f"[Camera/Keyframes] Request - session: {request.session_id}, num_keyframes: {len(request.keyframes)}")

        # Validate keyframes
        if len(request.keyframes) < 2:
            raise HTTPException(
                status_code=400,
                detail="Must provide at least 2 keyframes for interpolation"
            )

        # Validate identity parameters
        if not latent_code_path and request.seed is None:
            raise HTTPException(status_code=400, detail="Must provide either latent_code_path or seed")

        if latent_code_path and not os.path.exists(latent_code_path):
            raise HTTPException(status_code=404, detail=f"Latent code not found: {latent_code_path}")

        # Validate interpolation mode
        if request.interpolation not in ['linear', 'cubic']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interpolation '{request.interpolation}'. Must be 'linear' or 'cubic'"
            )

        # Create output directory
        output_dir = OUTPUTS_DIR / request.session_id / "camera_pose"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get camera pose service
        pose_service = get_camera_pose_service(MODEL_PATH)

        # Convert keyframes to trajectory via interpolation
        keyframes_dict = [
            {
                'timestamp': kf.timestamp,
                'yaw': kf.yaw,
                'pitch': kf.pitch,
                'radius': kf.radius,
            }
            for kf in request.keyframes
        ]

        trajectory = pose_service.interpolate_keyframes(
            keyframes=keyframes_dict,
            num_frames=request.num_frames,
            interpolation=request.interpolation,
        )

        print(f"[Camera/Keyframes] Interpolated trajectory with {len(trajectory)} frames from {len(request.keyframes)} keyframes")

        # Generate video
        video_path = await pose_service.generate_pose_video(
            session_id=request.session_id,
            output_dir=output_dir,
            trajectory=trajectory,
            latent_code_path=latent_code_path,
            generator_path=generator_path,
            seed=request.seed,
            truncation=request.truncation,
            num_frames=request.num_frames,
            progress_callback=None,  # TODO: Add WebSocket progress
        )

        print(f"[Camera/Keyframes] Video generated: {video_path}")

        # Generate URL (for now, returning trajectory file as placeholder)
        video_filename = Path(video_path).name
        video_url = f"/api/media/{request.session_id}/camera_pose/{video_filename}"

        return VideoGenerationResponse(
            success=True,
            session_id=request.session_id,
            video_url=video_url,
            message=f"Keyframe video generated (placeholder - actual video generation not yet implemented)",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return VideoGenerationResponse(
            success=False,
            session_id=request.session_id,
            message="Failed to generate keyframe video",
            error=str(e),
        )


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a mesh generation or video generation task
    """
    # TODO: Implement task status tracking
    return {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Task status tracking not yet implemented",
    }
