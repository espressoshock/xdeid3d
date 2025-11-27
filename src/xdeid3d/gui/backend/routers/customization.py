"""
Customization API routes for background alteration
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
import os
import uuid
from pathlib import Path
import sys

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.background_service import get_background_service

router = APIRouter()

# Configuration
MODEL_PATH = os.getenv('SPHEREHEAD_MODEL_PATH', str(Path(__file__).parent.parent.parent.parent / 'models' / 'spherehead-ckpt-025000.pkl'))
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

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

# Request models
class BackgroundColorRequest(BaseModel):
    session_id: str
    latent_code_path: Optional[str] = None  # For PTI-based identities
    seed: Optional[int] = None  # For seed-based identities
    truncation: Optional[float] = 0.7  # Truncation used in generation
    color: str  # Hex color
    generator_path: Optional[str] = None


class BackgroundTextRequest(BaseModel):
    session_id: str
    latent_code_path: Optional[str] = None  # For PTI-based identities
    seed: Optional[int] = None  # For seed-based identities
    truncation: Optional[float] = 0.7  # Truncation used in generation
    prompt: str
    negative_prompt: Optional[str] = ""
    generator_path: Optional[str] = None


# Response models
class CustomizationResponse(BaseModel):
    success: bool
    session_id: str
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    message: str
    error: Optional[str] = None


@router.post("/background/color", response_model=CustomizationResponse)
async def apply_color_background(request: BackgroundColorRequest):
    """
    Apply a solid color background to the generated identity

    This will:
    1. Load the latent code from Stage 1
    2. Re-synthesize the head with a solid color background
    3. Return the result image/video
    """
    try:
        # Convert URL paths to filesystem paths
        latent_code_path = url_to_filesystem_path(request.latent_code_path)
        generator_path = url_to_filesystem_path(request.generator_path)

        print(f"[Customization] Request - session: {request.session_id}, latent_path: {latent_code_path}, seed: {request.seed}, truncation: {request.truncation}")

        # Validate that we have either latent_code_path or seed
        if not latent_code_path and request.seed is None:
            raise HTTPException(status_code=400, detail="Must provide either latent_code_path (for PTI) or seed (for seed-based generation)")

        # Validate latent code path if provided
        if latent_code_path and not os.path.exists(latent_code_path):
            print(f"[Customization] ERROR: Latent code file not found at: {latent_code_path}")
            raise HTTPException(status_code=404, detail=f"Latent code not found: {latent_code_path}")

        # Parse hex color to RGB tuple
        hex_color = request.color.lstrip('#')
        if len(hex_color) != 6:
            raise HTTPException(status_code=400, detail="Invalid hex color format. Use #RRGGBB")

        try:
            bg_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid hex color value")

        # Create output directory
        output_dir = Path(__file__).parent.parent / "outputs" / request.session_id / "customization"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get background service
        bg_service = get_background_service(MODEL_PATH)

        # Get latent code: either load from file or generate from seed
        if latent_code_path:
            # PTI-based: load latent code
            ws = bg_service.load_latent_code(
                latent_code_path,
                generator_path
            )
        else:
            # Seed-based: generate latent code from seed
            ws = bg_service.generate_latent_from_seed(
                request.seed,
                truncation_psi=request.truncation
            )

        # Apply solid color background
        result_image = bg_service.apply_solid_background(
            ws,
            bg_color,
            truncation_psi=request.truncation
        )

        # Save result
        output_path = output_dir / "color_background.png"
        result_image.save(output_path, quality=95)

        # Generate relative URL for frontend
        image_url = f"/api/media/{request.session_id}/customization/color_background.png"

        return CustomizationResponse(
            success=True,
            session_id=request.session_id,
            image_url=image_url,
            message=f"Color background ({request.color}) applied successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return CustomizationResponse(
            success=False,
            session_id=request.session_id,
            message="Failed to apply color background",
            error=str(e),
        )


@router.post("/background/image", response_model=CustomizationResponse)
async def apply_image_background(
    session_id: str = Form(...),
    latent_code_path: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    truncation: float = Form(0.7),
    image_fit: str = Form("cover"),
    generator_path: Optional[str] = Form(None),
    background_image: UploadFile = File(...),
):
    """
    Apply a custom image as background

    This will:
    1. Upload and validate the background image
    2. Load the latent code from Stage 1
    3. Re-synthesize the head composited onto the background image
    4. Return the result
    """
    try:
        # Convert URL paths to filesystem paths
        latent_code_path_fs = url_to_filesystem_path(latent_code_path)
        generator_path_fs = url_to_filesystem_path(generator_path)

        print(f"[Customization/Image] Request - session: {session_id}, latent_path: {latent_code_path_fs}, seed: {seed}, truncation: {truncation}")

        # Validate that we have either latent_code_path or seed
        if not latent_code_path_fs and seed is None:
            raise HTTPException(status_code=400, detail="Must provide either latent_code_path or seed")

        # Validate latent code path if provided
        if latent_code_path_fs and not os.path.exists(latent_code_path_fs):
            print(f"[Customization/Image] ERROR: Latent code file not found at: {latent_code_path_fs}")
            raise HTTPException(status_code=404, detail=f"Latent code not found: {latent_code_path_fs}")

        if image_fit not in ['cover', 'contain', 'stretch']:
            raise HTTPException(status_code=400, detail="image_fit must be 'cover', 'contain', or 'stretch'")

        # Create output directory
        output_dir = Path(__file__).parent.parent / "outputs" / session_id / "customization"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded background image
        bg_image_path = output_dir / f"bg_upload_{uuid.uuid4().hex[:8]}.png"
        with open(bg_image_path, "wb") as f:
            content = await background_image.read()
            f.write(content)

        # Get background service
        bg_service = get_background_service(MODEL_PATH)

        # Get latent code: either load from file or generate from seed
        if latent_code_path_fs:
            ws = bg_service.load_latent_code(latent_code_path_fs, generator_path_fs)
        else:
            ws = bg_service.generate_latent_from_seed(seed, truncation_psi=truncation)

        # Apply image background
        result_image = bg_service.apply_image_background(
            ws,
            str(bg_image_path),
            fit_mode=image_fit,
            truncation_psi=truncation
        )

        # Save result
        output_path = output_dir / "image_background.png"
        result_image.save(output_path, quality=95)

        # Generate relative URL
        image_url = f"/api/media/{session_id}/customization/image_background.png"

        return CustomizationResponse(
            success=True,
            session_id=session_id,
            image_url=image_url,
            message=f"Background image applied successfully with {image_fit} fit",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return CustomizationResponse(
            success=False,
            session_id=session_id,
            message="Failed to apply image background",
            error=str(e),
        )


@router.post("/background/text", response_model=CustomizationResponse)
async def apply_text_background(request: BackgroundTextRequest):
    """
    Generate background from text prompt using Stable Diffusion

    This will:
    1. Generate a background image from the text prompt using SD
    2. Load the latent code from Stage 1
    3. Composite the head onto the generated background
    4. Return the result
    """
    try:
        # Convert URL paths to filesystem paths
        latent_code_path = url_to_filesystem_path(request.latent_code_path)
        generator_path = url_to_filesystem_path(request.generator_path)

        print(f"[Customization/Text] Request - session: {request.session_id}, latent_path: {latent_code_path}, seed: {request.seed}, truncation: {request.truncation}")

        # Validate that we have either latent_code_path or seed
        if not latent_code_path and request.seed is None:
            raise HTTPException(status_code=400, detail="Must provide either latent_code_path or seed")

        # Validate latent code path if provided
        if latent_code_path and not os.path.exists(latent_code_path):
            print(f"[Customization/Text] ERROR: Latent code file not found at: {latent_code_path}")
            raise HTTPException(status_code=404, detail=f"Latent code not found: {latent_code_path}")

        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Create output directory
        output_dir = Path(__file__).parent.parent / "outputs" / request.session_id / "customization"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get background service
        bg_service = get_background_service(MODEL_PATH)

        # Get latent code: either load from file or generate from seed
        if latent_code_path:
            ws = bg_service.load_latent_code(latent_code_path, generator_path)
        else:
            ws = bg_service.generate_latent_from_seed(request.seed, truncation_psi=request.truncation)

        # Generate and apply text background
        result_image, bg_path = bg_service.apply_text_background(
            ws,
            request.prompt,
            request.negative_prompt or "",
            output_dir=output_dir,
            truncation_psi=request.truncation
        )

        # Save result
        output_path = output_dir / "text_background.png"
        result_image.save(output_path, quality=95)

        # Generate relative URL
        image_url = f"/api/media/{request.session_id}/customization/text_background.png"

        return CustomizationResponse(
            success=True,
            session_id=request.session_id,
            image_url=image_url,
            message=f"Background generated from prompt: {request.prompt[:50]}...",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return CustomizationResponse(
            success=False,
            session_id=request.session_id,
            message="Failed to generate text background",
            error=str(e),
        )


@router.get("/status/{session_id}")
async def get_customization_status(session_id: str):
    """
    Get the status of a customization job
    """
    # TODO: Implement job status tracking
    return {
        "session_id": session_id,
        "status": "idle",
        "progress": 0.0,
    }
