import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from models.requests import SeedGenerationRequest, TextGenerationRequest
from models.responses import ApiResponse, UploadResponse
from services.spherehead_service import spherehead_service
from utils.image_processing import validate_image, save_upload

router = APIRouter()

UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


@router.post("/generation/seed", response_model=ApiResponse)
async def generate_from_seed(request: SeedGenerationRequest, background_tasks: BackgroundTasks):
    """
    Initiate seed-based generation. Progress will be sent via WebSocket.
    """
    try:
        # Enqueue generation task
        # The actual generation will communicate via WebSocket
        return ApiResponse(
            success=True,
            message="Generation started",
            data={"session_id": request.session_id},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generation/upload", response_model=ApiResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for PTI projection.
    """
    try:
        # Validate image
        content = await file.read()
        if not validate_image(content):
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Generate unique ID
        upload_id = str(uuid.uuid4())
        filename = f"{upload_id}_{file.filename}"
        filepath = UPLOADS_DIR / filename

        # Save file
        await save_upload(content, filepath)

        return ApiResponse(
            success=True,
            message="File uploaded successfully",
            data={
                "upload_id": upload_id,
                "filename": filename,
                "size": len(content),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generation/text-to-image", response_model=ApiResponse)
async def generate_from_text(request: TextGenerationRequest, background_tasks: BackgroundTasks):
    """
    Initiate text-to-image generation followed by PTI. Progress via WebSocket.
    """
    try:
        return ApiResponse(
            success=True,
            message="Text generation started",
            data={"session_id": request.session_id},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
