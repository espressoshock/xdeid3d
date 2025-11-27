"""
PTI Service - Subprocess-based wrapper for PTI (Pivotal Tuning Inversion) projection
"""
import os
import asyncio
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class PTIService:
    """Service for running PTI projection via subprocess"""

    def __init__(self):
        # Paths relative to backend directory
        self.backend_dir = Path(__file__).parent.parent
        self.core_dir = self.backend_dir.parent.parent / "core"
        self.model_path = self.backend_dir.parent.parent / "models" / "spherehead-ckpt-025000.pkl"
        self.conda_env = "xdeid3d"  # Correct conda environment for X-DeID3D

        # Verify core directory exists (required)
        if not self.core_dir.exists():
            raise RuntimeError(f"Core directory not found: {self.core_dir}")

    async def project_image(
        self,
        session_id: str,
        upload_id: str,
        w_steps: int = 500,
        pti_steps: int = 350,
        truncation: float = 0.7,
        nrr: int = 128,
        sample_mult: float = 1.5,
        generate_video: bool = True,
        optimize_noise: bool = False,
        initial_noise_factor: float = 0.05,
        noise_ramp_length: float = 0.75,
        regularize_noise_weight: float = 1e5,
        websocket=None,
    ) -> Dict[str, Any]:
        """
        Project uploaded image to latent space and fine-tune generator

        Args:
            session_id: Unique session identifier
            upload_id: ID of uploaded image
            w_steps: Steps for W-projection (default: 500)
            pti_steps: Steps for PTI fine-tuning (default: 350)
            truncation: Truncation psi for video generation
            nrr: Neural rendering resolution
            sample_mult: Depth sampling multiplier
            generate_video: Whether to generate 360° video after PTI
            optimize_noise: Enable noise optimization for finer details
            initial_noise_factor: Initial noise injection strength (default: 0.05)
            noise_ramp_length: Fraction of steps with noise active (default: 0.75)
            regularize_noise_weight: Noise smoothness constraint (default: 1e5)
            websocket: WebSocket connection for progress updates

        Returns:
            Dict with image_url, latent paths, etc.
        """
        # Check if model exists
        if not self.model_path.exists():
            error_msg = f"Model file not found: {self.model_path}"
            print(f"[PTI] {error_msg}")
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "error": error_msg,
                    "code": "MODEL_NOT_FOUND"
                })
            raise FileNotFoundError(error_msg)

        # Find uploaded image
        uploads_dir = self.backend_dir / "uploads"
        uploaded_files = list(uploads_dir.glob(f"{upload_id}_*"))
        if not uploaded_files:
            error_msg = f"Uploaded file not found for upload_id: {upload_id}"
            print(f"[PTI] {error_msg}")
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "error": error_msg,
                    "code": "FILE_NOT_FOUND"
                })
            raise FileNotFoundError(error_msg)

        uploaded_file = uploaded_files[0]

        # Create output directory for this session
        output_dir = self.backend_dir / "outputs" / session_id
        pti_output_dir = output_dir / "pti"
        pti_output_dir.mkdir(parents=True, exist_ok=True)

        # Create target directory with uploaded image
        target_dir = pti_output_dir / "target"
        target_dir.mkdir(exist_ok=True)
        target_image = target_dir / "0.jpg"

        # Copy uploaded file to target directory
        import shutil
        shutil.copy(uploaded_file, target_image)

        # Create dataset.json with default camera parameters
        # These are standard frontal camera parameters for PTI
        self._create_dataset_json(target_dir, "0.jpg")

        # Use absolute paths for subprocess
        model_path_abs = str(self.model_path.absolute())
        pti_output_abs = str(pti_output_dir.absolute())
        target_dir_abs = str(target_dir.absolute())

        print(f"[PTI] Starting PTI projection for upload_id {upload_id}")
        print(f"[PTI] W-steps: {w_steps}, PTI-steps: {pti_steps}")
        print(f"[PTI] Target image: {target_image}")

        # Setup progress file for real-time monitoring
        progress_file = output_dir / "pti_progress.json"

        # Setup preview directory for real-time image updates
        preview_dir = pti_output_dir / "preview"
        preview_dir.mkdir(exist_ok=True)

        # Clean up any old preview images from previous runs
        old_preview = preview_dir / "preview_current.jpg"
        if old_preview.exists():
            old_preview.unlink()
            print(f"[PTI] Cleaned up old preview image")

        try:
            # Stage 1: Run projector.py (W-projection + PTI tuning)
            await self._run_projector(
                model_path_abs,
                target_dir_abs,
                pti_output_abs,
                w_steps,
                pti_steps,
                progress_file,
                preview_dir,
                optimize_noise,
                initial_noise_factor,
                noise_ramp_length,
                regularize_noise_weight,
                websocket,
                session_id
            )

            # Find generated files
            # projector.py outputs to: {outdir}/{model_name}/{image_name}/
            model_name = "spherehead-ckpt-025000.pkl"
            result_dir = pti_output_dir / model_name / "0.jpg"

            if not result_dir.exists():
                raise RuntimeError(f"PTI output directory not found: {result_dir}")

            # Check for essential files
            projected_w_path = result_dir / "projected_w.npz"
            finetuned_gen_path = result_dir / "fintuned_generator.pkl"
            proj_image_path = result_dir / "proj.png"

            if not projected_w_path.exists():
                raise RuntimeError(f"Projected latent not found: {projected_w_path}")
            if not finetuned_gen_path.exists():
                raise RuntimeError(f"Fine-tuned generator not found: {finetuned_gen_path}")

            # Copy projection result to output directory for easier access
            result_image_path = output_dir / "pti_result.png"
            if proj_image_path.exists():
                import shutil
                shutil.copy(proj_image_path, result_image_path)
                print(f"[PTI] Copied projection result to {result_image_path}")

            # Copy progress video if it exists (proj.mp4 shows optimization progression)
            proj_video_path = result_dir / "proj.mp4"
            progress_video_path = output_dir / "pti_progress.mp4"
            if proj_video_path.exists():
                import shutil
                shutil.copy(proj_video_path, progress_video_path)
                print(f"[PTI] Copied progress video to {progress_video_path}")

            result = {
                "imageUrl": f"/api/outputs/{session_id}/pti_result.png",
                "progressVideoUrl": f"/api/outputs/{session_id}/pti_progress.mp4" if proj_video_path.exists() else None,
                "latentCodePath": f"/api/outputs/{session_id}/pti/{model_name}/0.jpg/projected_w.npz",
                "generatorPath": f"/api/outputs/{session_id}/pti/{model_name}/0.jpg/fintuned_generator.pkl",
                "metadata": {
                    "upload_id": upload_id,
                    "method": "upload",
                    "w_steps": w_steps,
                    "pti_steps": pti_steps,
                    "timestamp": datetime.now().isoformat(),
                }
            }

            # Stage 2 (optional): Generate 360° video
            if generate_video:
                await self._generate_pti_video(
                    finetuned_gen_path,
                    projected_w_path,
                    output_dir,
                    truncation,
                    nrr,
                    websocket
                )

                # Update result with video URL
                video_path = output_dir / "video.mp4"
                if video_path.exists():
                    result["imageUrl"] = f"/api/outputs/{session_id}/video.mp4"

            # Save metadata
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(result["metadata"], f, indent=2)

            # Send completion message
            if websocket:
                await websocket.send_json({
                    "type": "complete",
                    "result": result
                })

            print(f"[PTI] Projection complete for upload_id {upload_id}")
            return result

        except Exception as e:
            error_msg = f"PTI projection error: {str(e)}"
            print(f"[PTI] {error_msg}")
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "error": error_msg,
                    "code": "PTI_ERROR"
                })
            raise

    async def _run_projector(
        self,
        model_path: str,
        target_dir: str,
        output_dir: str,
        w_steps: int,
        pti_steps: int,
        progress_file: Path,
        preview_dir: Path,
        optimize_noise: bool,
        initial_noise_factor: float,
        noise_ramp_length: float,
        regularize_noise_weight: float,
        websocket,
        session_id: str
    ):
        """Run projector.py subprocess with real-time preview image generation"""

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", "-u", "projector.py",
            "--outdir", output_dir,
            "--target", target_dir,
            "--network", model_path,
            "--idx", "0",
            "--filename", "0.jpg",
            "--num-steps", str(w_steps),
            "--num-steps-pti", str(pti_steps),
            "--save-video", "true",  # Generate optimization progress video (proj.mp4)
            "--progress-file", str(progress_file.absolute()),
            "--preview-dir", str(preview_dir.absolute()),  # Enable real-time preview images
            "--preview-interval", "10",  # Save preview every 10 steps
            "--optimize-noise", str(optimize_noise).lower(),
        ]

        # Add noise-specific parameters only if noise optimization is enabled
        if optimize_noise:
            cmd.extend([
                "--initial-noise-factor", str(initial_noise_factor),
                "--noise-ramp-length", str(noise_ramp_length),
                "--regularize-noise-weight", str(int(regularize_noise_weight)),
            ])

        print(f"[PTI] Running projector.py...")
        print(f"[PTI] Command: {' '.join(cmd)}")

        if websocket:
            await websocket.send_json({
                "type": "progress",
                "stage": "w_projection",
                "progress": 0.0,
                "step": 0,
                "total_steps": w_steps + pti_steps,
                "message": "Initializing PTI projection..."
            })

        # Start subprocess
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1"})

        print(f"[PTI] Starting subprocess with progress file: {progress_file}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.core_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Monitor progress file and preview images in parallel with output logging
        progress_task = asyncio.create_task(
            self._monitor_progress_file(progress_file, w_steps, pti_steps, websocket, preview_dir, session_id)
        )

        # Log stdout/stderr for debugging
        log_task = asyncio.create_task(
            self._log_output(process)
        )

        # Wait for completion
        returncode = await process.wait()

        # Cancel progress monitoring
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        if returncode != 0:
            raise RuntimeError(f"Projector failed with return code {returncode}")

    async def _monitor_progress_file(
        self,
        progress_file: Path,
        w_steps: int,
        pti_steps: int,
        websocket,
        preview_dir: Path,
        session_id: str
    ):
        """
        Monitor progress file AND preview images for real-time updates

        This enhanced monitor watches both:
        1. Progress JSON file for step/stage updates
        2. Preview image file for visual updates

        When either updates, it sends a WebSocket message with both progress and preview URL.
        """
        print(f"[PTI Progress Monitor] Starting monitoring of {progress_file}")
        print(f"[PTI Progress Monitor] Monitoring preview images in {preview_dir}")

        last_progress = {"current": 0, "total": w_steps, "stage": "w_projection"}
        last_preview_mtime = 0
        last_preview_url = None  # Track the last preview URL
        check_interval = 0.1  # Check every 100ms
        total_steps = w_steps + pti_steps

        # Preview image path (single file overwritten each time)
        preview_path = preview_dir / "preview_current.jpg"

        # Check if preview already exists before starting monitoring
        if preview_path.exists():
            last_preview_mtime = preview_path.stat().st_mtime
            last_preview_url = f"/api/outputs/{session_id}/pti/preview/preview_current.jpg?t={int(last_preview_mtime * 1000)}"
            print(f"[PTI Progress Monitor] Found existing preview: {last_preview_url}")

        while True:
            try:
                await asyncio.sleep(check_interval)

                # Check if progress file exists
                if not progress_file.exists():
                    continue

                # Read progress file
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)

                    # Check if preview image exists and update URL if modified
                    preview_updated = False
                    if preview_path.exists():
                        current_mtime = preview_path.stat().st_mtime
                        if current_mtime > last_preview_mtime:
                            last_preview_mtime = current_mtime
                            preview_updated = True
                            # Create URL with cache-busting timestamp
                            # Format: /api/outputs/{session_id}/pti/preview/preview_current.jpg?t={timestamp}
                            last_preview_url = f"/api/outputs/{session_id}/pti/preview/preview_current.jpg?t={int(current_mtime * 1000)}"
                            print(f"[PTI Progress Monitor] New preview image detected: {last_preview_url}")

                    # Send update if progress changed OR preview updated
                    if (progress_data["current"] != last_progress["current"] or
                        progress_data.get("stage") != last_progress.get("stage") or
                        preview_updated):

                        last_progress = progress_data

                        # Calculate overall progress accounting for both stages
                        stage = progress_data.get("stage", "w_projection")
                        current = progress_data["current"]
                        stage_total = progress_data["total"]

                        if stage == "w_projection":
                            overall_step = current
                        elif stage == "pti_tuning":
                            overall_step = w_steps + current
                        else:
                            overall_step = current

                        overall_progress = overall_step / total_steps

                        stage_label = "W-projection" if stage == "w_projection" else "PTI tuning"
                        message = f"{stage_label}: {current}/{stage_total}"

                        # Build WebSocket message
                        ws_message = {
                            "type": "progress",
                            "stage": stage,
                            "progress": overall_progress,
                            "step": overall_step,
                            "total_steps": total_steps,
                            "message": message
                        }

                        # ALWAYS include preview URL if available (even if not just updated)
                        if last_preview_url:
                            ws_message["preview_image"] = last_preview_url

                        # Send update to WebSocket
                        await websocket.send_json(ws_message)

                        log_msg = f"[PTI Progress] → UI: {stage_label} {current}/{stage_total} ({int(overall_progress*100)}%)"
                        if preview_updated:
                            log_msg += " [preview updated]"
                        if last_preview_url:
                            log_msg += f" [preview: {last_preview_url}]"
                        print(log_msg)

                except (json.JSONDecodeError, KeyError) as e:
                    # File might be partially written, skip this iteration
                    continue
                except Exception as e:
                    print(f"[PTI Progress Monitor] Error reading file: {e}")
                    continue

            except asyncio.CancelledError:
                print(f"[PTI Progress Monitor] Monitoring cancelled")
                break
            except Exception as e:
                print(f"[PTI Progress Monitor] Unexpected error: {e}")
                break

    async def _log_output(self, process):
        """Log projector.py output for debugging"""
        async def read_stream(stream, prefix):
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                try:
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"[projector.py] {line}")
                except Exception:
                    pass

        await asyncio.gather(
            read_stream(process.stdout, "stdout"),
            read_stream(process.stderr, "stderr"),
            return_exceptions=True
        )

    def _create_dataset_json(self, target_dir: Path, filename: str):
        """
        Create dataset.json with default camera parameters for PTI projection

        Camera parameters are based on SphereHead defaults:
        - Frontal view (yaw=pi/2, pitch=pi/2)
        - Radius=2.7
        - Intrinsics: focal_length=4.2647
        """
        # Default camera parameters for frontal view
        # cam2world matrix (4x4) + intrinsics (3x3) = 25 values
        camera_params = [
            # cam2world pose matrix (16 values - 4x4 flattened)
            1.0000923871994019,
            0.004083682782948017,
            -0.005159023217856884,
            0.013929363340139389,
            -0.024401208385825157,
            -0.9822011590003967,
            -0.189874067902565,
            0.512660026550293,
            -0.0005358866765163839,
            0.18985432386398315,
            -0.9821922779083252,
            2.651919364929199,
            0.0, 0.0, 0.0, 1.0,
            # intrinsics matrix (9 values - 3x3 flattened)
            4.2647, 0.0, 0.5,
            0.0, 4.2647, 0.5,
            0.0, 0.0, 1.0
        ]

        dataset_json = {
            "labels": [
                [filename, camera_params]
            ]
        }

        dataset_path = target_dir / "dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(dataset_json, f, indent=4)

        print(f"[PTI] Created dataset.json at {dataset_path}")

    async def _generate_pti_video(
        self,
        generator_path: Path,
        latent_path: Path,
        output_dir: Path,
        truncation: float,
        nrr: int,
        websocket
    ):
        """Generate 360° video from PTI results"""

        video_output = output_dir / "video.mp4"

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", "-u", "gen_videos_proj_withseg.py",
            "--output", str(video_output.absolute()),
            "--latent", str(latent_path.absolute()),
            "--network", str(generator_path.absolute()),
            "--trunc", str(truncation),
            "--nrr", str(nrr),
            "--cfg", "Head",
        ]

        print(f"[PTI] Generating video from PTI results...")

        if websocket:
            await websocket.send_json({
                "type": "progress",
                "stage": "synthesis",
                "progress": 0.9,
                "message": "Generating 360° video from PTI results..."
            })

        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1"})

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.core_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Simple monitoring (just log output)
        async def log_output(stream, prefix):
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"[gen_videos_proj] {line}")

        await asyncio.gather(
            log_output(process.stdout, "stdout"),
            log_output(process.stderr, "stderr"),
            return_exceptions=True
        )

        returncode = await process.wait()

        if returncode != 0:
            print(f"[PTI] Video generation failed with return code {returncode}")
            # Don't fail the whole PTI process, just skip video


# Global service instance
pti_service = PTIService()
