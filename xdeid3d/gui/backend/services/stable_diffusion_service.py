"""
Stable Diffusion Service - Text-to-image generation
"""
import torch
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from datetime import datetime

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class StableDiffusionService:
    """Service for Stable Diffusion text-to-image generation"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.model_id = "runwayml/stable-diffusion-v1-5"  # Lightweight, good for faces
        self.backend_dir = Path(__file__).parent.parent

        self._initialized = True

    def load_model(self):
        """Load Stable Diffusion model (lazy loading)"""
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers library not installed. Install with: pip install diffusers")

        if self.pipe is not None:
            return

        print(f"[StableDiffusion] Loading model: {self.model_id}")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        self.pipe = self.pipe.to(self.device)

        # Enable optimizations
        if self.device.type == "cuda":
            self.pipe.enable_attention_slicing()

        print("[StableDiffusion] Model loaded successfully")

    async def generate_face(
        self,
        session_id: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: int = 50,
        guidance_scale: float = 7.5,
        websocket = None,
    ) -> Dict[str, Any]:
        """
        Generate face image from text prompt

        Args:
            session_id: Unique session identifier
            prompt: Text description
            negative_prompt: What to avoid
            steps: Number of diffusion steps (20-100)
            guidance_scale: How closely to follow prompt (5-15)
            websocket: WebSocket connection for progress updates

        Returns:
            Dict with image_path, upload_id, and metadata
        """
        # Create session output directory
        output_dir = self.backend_dir / "outputs" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup progress tracking file
        progress_file = output_dir / "sd_progress.json"

        # Enhance prompt for face generation
        enhanced_prompt = f"professional portrait photo, {prompt}, centered face, high quality, detailed, 4k"

        if negative_prompt is None:
            negative_prompt = "blurry, low quality, distorted, multiple faces, bad anatomy, deformed"

        print(f"[StableDiffusion] Generating image for session {session_id}")
        print(f"[StableDiffusion] Prompt: {enhanced_prompt}")

        try:
            # Send initial progress
            await self._update_progress(websocket, progress_file, 0.0, 0, steps, "Loading model...")

            # Load model (lazy loading)
            self.load_model()

            await self._update_progress(websocket, progress_file, 0.05, 0, steps, "Model loaded, starting generation...")

            # Get the current event loop for callbacks
            loop = asyncio.get_event_loop()

            # Create progress callback for diffusion steps
            async def step_callback(step: int, timestep: int, latents):
                progress = 0.05 + (0.9 * step / steps)  # 5% to 95%
                message = f"Generating image... step {step}/{steps}"
                await self._update_progress(websocket, progress_file, progress, step, steps, message)

            # Generate image in thread pool to avoid blocking
            def generate_sync():
                # Custom callback wrapper for diffusers
                def callback_wrapper(step, timestep, latents):
                    # Schedule async callback in the main event loop from thread
                    if websocket and loop:
                        asyncio.run_coroutine_threadsafe(
                            step_callback(step, timestep, latents),
                            loop
                        )

                return self.pipe(
                    enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    callback=callback_wrapper if websocket else None,
                    callback_steps=1,
                ).images[0]

            # Run generation in executor to prevent blocking
            image = await loop.run_in_executor(None, generate_sync)

            await self._update_progress(websocket, progress_file, 0.95, steps, steps, "Saving image...")

            # Save generated image to outputs directory
            image_filename = "sd_generated.png"
            image_path = output_dir / image_filename
            image.save(str(image_path))

            # Also save to uploads directory for PTI pipeline
            # PTI expects: uploads/{upload_id}_{filename} pattern
            uploads_dir = self.backend_dir / "uploads"
            uploads_dir.mkdir(exist_ok=True)
            upload_filename = f"{session_id}_{image_filename}"
            upload_path = uploads_dir / upload_filename
            image.save(str(upload_path))

            print(f"[StableDiffusion] Image saved to {image_path}")
            print(f"[StableDiffusion] Upload copy saved to {upload_path}")

            await self._update_progress(websocket, progress_file, 1.0, steps, steps, "Image generation complete!")

            return {
                "image_path": str(image_path),
                "upload_id": session_id,  # Use session_id as upload_id for PTI
                "upload_path": str(upload_path),
                "metadata": {
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "negative_prompt": negative_prompt,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "timestamp": datetime.now().isoformat(),
                },
            }

        except Exception as e:
            error_msg = f"Stable Diffusion generation failed: {str(e)}"
            print(f"[StableDiffusion] Error: {error_msg}")

            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "error": error_msg,
                    "code": "SD_GENERATION_ERROR"
                })

            raise

    async def _update_progress(self, websocket, progress_file: Path, progress: float, step: int, total_steps: int, message: str):
        """Update progress via WebSocket and file"""
        progress_data = {
            "progress": progress,
            "step": step,
            "total_steps": total_steps,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        # Write to file for monitoring
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f)
        except Exception as e:
            print(f"[StableDiffusion] Failed to write progress file: {e}")

        # Send via WebSocket
        if websocket:
            try:
                await websocket.send_json({
                    "type": "progress",
                    "stage": "sd_generation",
                    "progress": progress,
                    "step": step,
                    "total_steps": total_steps,
                    "message": message,
                })
            except Exception as e:
                print(f"[StableDiffusion] Failed to send WebSocket progress: {e}")


# Global singleton instance
stable_diffusion_service = StableDiffusionService()
