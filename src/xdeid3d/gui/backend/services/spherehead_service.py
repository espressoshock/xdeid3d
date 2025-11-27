"""
SphereHead Service - Subprocess-based wrapper for X-DeID3D core generation
"""
import os
import asyncio
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class SphereHeadService:
    """Service for running SphereHead generation via subprocess"""

    def __init__(self):
        # Paths relative to backend directory
        self.backend_dir = Path(__file__).parent.parent
        self.core_dir = self.backend_dir.parent.parent / "core"
        self.model_path = self.backend_dir.parent.parent / "models" / "spherehead-ckpt-025000.pkl"
        self.conda_env = "xdeid3d"  # Correct conda environment for X-DeID3D

        # Verify core directory exists (required)
        if not self.core_dir.exists():
            raise RuntimeError(f"Core directory not found: {self.core_dir}")

        # Model check is deferred to generation time (allows server to start without model)

    async def generate_from_seed(
        self,
        session_id: str,
        seed: int,
        truncation: float = 0.7,
        nrr: int = 128,  # Not used by gen_samples.py, kept for compatibility
        sample_mult: float = 1.5,  # Not used by gen_samples.py, kept for compatibility
        websocket=None,
    ) -> Dict[str, Any]:
        """
        Generate single identity image from seed using gen_samples.py

        Args:
            session_id: Unique session identifier
            seed: Random seed for generation
            truncation: Truncation psi (0.5-0.7)
            nrr: Not used for single image generation
            sample_mult: Not used for single image generation
            websocket: WebSocket connection for progress updates

        Returns:
            Dict with image_url, depth_url, metadata, etc.
        """
        # Check if model exists
        if not self.model_path.exists():
            error_msg = f"Model file not found: {self.model_path}. Please download checkpoint-025000.pkl to the models/ directory."
            print(f"[SphereHead] {error_msg}")
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "error": error_msg,
                    "code": "MODEL_NOT_FOUND"
                })
            raise FileNotFoundError(error_msg)

        # Create output directory for this session
        output_dir = self.backend_dir / "outputs" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use absolute paths for subprocess
        model_path_abs = str(self.model_path.absolute())
        output_dir_abs = str(output_dir.absolute())

        # Use gen_single_sample.py to generate ONLY a frontal identity image (not multi-angle grid)
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", "-u", "gen_single_sample.py",
            "--network", model_path_abs,
            "--seed", str(seed),
            "--outdir", output_dir_abs,
            "--trunc", str(truncation),
        ]

        print(f"[SphereHead] Starting frontal identity generation for seed {seed}")
        print(f"[SphereHead] Command: {' '.join(cmd)}")
        print(f"[SphereHead] Working directory: {self.core_dir}")

        try:
            # Send initial progress
            if websocket:
                await websocket.send_json({
                    "type": "progress",
                    "stage": "synthesis",
                    "progress": 0.1,
                    "step": 0,
                    "total_steps": 1,
                    "message": f"Generating identity image for seed {seed}..."
                })

            # Prepare environment
            import subprocess

            env = os.environ.copy()
            env.update({
                "PYTHONUNBUFFERED": "1",
            })

            print(f"[SphereHead] Starting gen_samples.py subprocess")

            # Start subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.core_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Monitor stdout/stderr
            stdout_task = asyncio.create_task(
                self._monitor_async_output(process, websocket)
            )

            # Wait for process to complete
            returncode = await process.wait()

            if returncode != 0:
                error_msg = f"Generation failed with return code {returncode}"
                print(f"[SphereHead] {error_msg}")
                if websocket:
                    await websocket.send_json({
                        "type": "error",
                        "error": error_msg,
                        "code": "GENERATION_FAILED"
                    })
                raise RuntimeError(error_msg)

            # Find generated image file (gen_single_sample.py outputs: {outdir}/seed####.png)
            seed_str = f"seed{seed:04d}"
            image_file = output_dir / f"{seed_str}.png"

            if not image_file.exists():
                error_msg = f"Generated image file not found: {image_file}"
                print(f"[SphereHead] {error_msg}")
                print(f"[SphereHead] Directory contents: {list(output_dir.iterdir())}")
                raise FileNotFoundError(error_msg)

            print(f"[SphereHead] Found generated frontal image: {image_file}")

            # Send completion progress
            if websocket:
                await websocket.send_json({
                    "type": "progress",
                    "stage": "synthesis",
                    "progress": 1.0,
                    "step": 1,
                    "total_steps": 1,
                    "message": "Identity image generated successfully!"
                })

            # Build result URL (direct path, no subdirectory)
            # image_file is: outputs/session_id/seed0046.png
            # URL: /api/media/session_id/seed0046.png
            result = {
                "imageUrl": f"/api/media/{session_id}/{image_file.name}",
                "depthUrl": None,  # Single image generation doesn't produce depth
                "latentCodePath": None,  # Not applicable for seed generation
                "metadata": {
                    "seed": seed,
                    "method": "seed",
                    "truncation": truncation,
                    "timestamp": datetime.now().isoformat(),
                },
            }

            # Send completion message
            if websocket:
                await websocket.send_json({
                    "type": "complete",
                    "result": result
                })

            print(f"[SphereHead] Generation complete for seed {seed}")
            return result

        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            print(f"[SphereHead] {error_msg}")
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "error": error_msg,
                    "code": "GENERATION_ERROR"
                })
            raise

    async def _monitor_progress_file(self, progress_file: Path, websocket):
        """Monitor progress file for real-time updates"""
        import time

        print(f"[Progress File Monitor] Starting monitoring of {progress_file}")

        last_progress = {"current": 0, "total": 240}
        check_interval = 0.1  # Check every 100ms

        while True:
            try:
                await asyncio.sleep(check_interval)

                # Check if file exists
                if not progress_file.exists():
                    continue

                # Read progress file
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)

                    # Check if progress changed
                    if progress_data["current"] != last_progress["current"]:
                        last_progress = progress_data

                        # Send update to WebSocket
                        await websocket.send_json({
                            "type": "progress",
                            "stage": "synthesis",
                            "progress": progress_data["progress"],
                            "step": progress_data["current"],
                            "total_steps": progress_data["total"],
                            "message": f"Generating frame {progress_data['current']}/{progress_data['total']}"
                        })

                        print(f"[Progress File] → UI: {progress_data['current']}/{progress_data['total']} ({int(progress_data['progress']*100)}%)")

                except (json.JSONDecodeError, KeyError) as e:
                    # File might be partially written, skip this iteration
                    continue
                except Exception as e:
                    print(f"[Progress File Monitor] Error reading file: {e}")
                    continue

            except asyncio.CancelledError:
                print(f"[Progress File Monitor] Monitoring cancelled")
                break
            except Exception as e:
                print(f"[Progress File Monitor] Unexpected error: {e}")
                break

    async def _monitor_async_output(self, process, websocket):
        """Monitor asyncio subprocess output in real-time"""
        import time

        total_frames = 240
        line_count = 0

        print(f"[Output Monitor] Starting real-time output monitoring...")

        try:
            while True:
                # Read line from stdout (non-blocking, async)
                line_bytes = await process.stdout.readline()

                if not line_bytes:
                    # EOF reached, process finished
                    break

                # Decode line
                try:
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    if line:
                        line_count += 1
                        # Process the line
                        await self._process_output_line(line, total_frames, websocket)

                        # CRITICAL: Yield after every line to give event loop a chance to send WebSocket messages
                        # This is necessary because readline() can return very fast, and without yielding,
                        # the WebSocket send buffer never gets flushed
                        await asyncio.sleep(0.001)  # 1ms sleep to force event loop to run

                except Exception as e:
                    print(f"[Output Monitor] Error processing line: {e}")
                    await asyncio.sleep(0.001)  # Yield even on error
                    continue

        except Exception as e:
            print(f"[Output Monitor] Error in monitoring: {e}")
            import traceback
            traceback.print_exc()

        print(f"[Output Monitor] Monitoring complete (processed {line_count} lines)")

    async def _monitor_pty_output_OLD(self, master_fd, process, websocket):
        """Monitor PTY output in real-time and parse progress"""
        import time
        import select

        buffer = b''
        total_frames = 240
        last_update = time.time()

        print(f"[PTY Monitor] Starting real-time output monitoring...")

        # Track if we've seen any output
        output_count = 0

        while True:
            # Use select to check if data is available (non-blocking with timeout)
            try:
                ready, _, _ = select.select([master_fd], [], [], 0.05)

                if ready:
                    # Data is available, read it
                    try:
                        chunk = os.read(master_fd, 4096)
                        if chunk:
                            output_count += 1
                            buffer += chunk

                            # Process buffer line by line
                            while b'\n' in buffer or b'\r' in buffer:
                                # Find next line terminator
                                pos_n = buffer.find(b'\n')
                                pos_r = buffer.find(b'\r')

                                if pos_n == -1:
                                    pos = pos_r
                                elif pos_r == -1:
                                    pos = pos_n
                                else:
                                    pos = min(pos_n, pos_r)

                                # Extract line
                                line = buffer[:pos]
                                buffer = buffer[pos+1:]

                                # Decode and process
                                try:
                                    line_str = line.decode('utf-8', errors='ignore').strip()
                                    if line_str:
                                        await self._process_output_line(line_str, total_frames, websocket)
                                except Exception as e:
                                    print(f"[PTY Monitor] Error processing line: {e}")

                    except (OSError, IOError) as e:
                        # Error reading, check if process exited
                        if process.poll() is not None:
                            break
                else:
                    # No data available, check if process exited
                    if process.poll() is not None:
                        # Process exited, do final read of any remaining data
                        print(f"[PTY Monitor] Process exited with code {process.returncode}")
                        try:
                            # Set non-blocking and read all remaining data
                            while True:
                                chunk = os.read(master_fd, 4096)
                                if not chunk:
                                    break
                                buffer += chunk
                                output_count += 1

                                # Process any remaining lines
                                while b'\n' in buffer or b'\r' in buffer:
                                    pos_n = buffer.find(b'\n')
                                    pos_r = buffer.find(b'\r')
                                    if pos_n == -1:
                                        pos = pos_r
                                    elif pos_r == -1:
                                        pos = pos_n
                                    else:
                                        pos = min(pos_n, pos_r)
                                    line = buffer[:pos]
                                    buffer = buffer[pos+1:]
                                    try:
                                        line_str = line.decode('utf-8', errors='ignore').strip()
                                        if line_str:
                                            await self._process_output_line(line_str, total_frames, websocket)
                                    except Exception as e:
                                        print(f"[PTY Monitor] Error processing final line: {e}")
                        except (OSError, IOError):
                            pass
                        break

                # Small yield to event loop
                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"[PTY Monitor] Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                if process.poll() is not None:
                    break

        print(f"[PTY Monitor] Monitoring complete (read {output_count} chunks)")

    async def _process_output_line(self, line_str, total_frames, websocket):
        """Process a single output line and send updates"""
        # Log the line (only non-empty lines)
        if line_str:
            print(f"[gen_videos.py] {line_str}")

        if not websocket:
            return

        # Parse tqdm progress bar format: look for patterns like "10%|███" or "50/240"
        # Pattern 1: "X/Y" format (e.g., "10/240" - frame counter)
        match = re.search(r'(\d+)/(\d+)', line_str)
        if match:
            current_frame = int(match.group(1))
            total = int(match.group(2))
            progress = current_frame / total

            await websocket.send_json({
                "type": "progress",
                "stage": "synthesis",
                "progress": progress,
                "step": current_frame,
                "total_steps": total,
                "message": f"Generating frame {current_frame}/{total}"
            })
            print(f"[Progress] → UI: {current_frame}/{total} ({int(progress*100)}%)")
            # Force immediate transmission by yielding to event loop
            await asyncio.sleep(0)
            return

        # Pattern 2: Percentage with bar: "XX%|" (tqdm format)
        match = re.search(r'(\d+)%\|', line_str)
        if match:
            percent = int(match.group(1))
            progress = percent / 100.0

            # Extract ETA if present
            eta_match = re.search(r'(\d+:\d+)', line_str)
            eta = eta_match.group(1) if eta_match else ""

            message = f"Generating... {percent}%"
            if eta:
                message += f" (ETA: {eta})"

            await websocket.send_json({
                "type": "progress",
                "stage": "synthesis",
                "progress": progress,
                "step": percent,
                "total_steps": 100,
                "message": message
            })
            print(f"[Progress] → UI: {percent}%")
            return

        # Pattern 3: Simple percentage at start of line
        match = re.search(r'^\s*(\d+)%', line_str)
        if match:
            percent = int(match.group(1))
            progress = percent / 100.0

            await websocket.send_json({
                "type": "progress",
                "stage": "synthesis",
                "progress": progress,
                "step": percent,
                "total_steps": 100,
                "message": f"Processing... {percent}%"
            })
            return

        # Status messages
        if "Loading networks" in line_str or "Loading network" in line_str:
            await websocket.send_json({
                "type": "progress",
                "stage": "synthesis",
                "progress": 0.05,
                "message": "Loading SphereHead model..."
            })
        elif "Generating" in line_str and "video" in line_str:
            await websocket.send_json({
                "type": "progress",
                "stage": "synthesis",
                "progress": 0.1,
                "message": "Starting video generation..."
            })

    async def _monitor_progress_OLD(self, process, websocket):
        """Monitor subprocess output and send progress updates"""

        async def read_stream(stream, prefix):
            """Read from a stream and log output, handling both \\n and \\r"""
            import time
            line_count = 0
            buffer = b''

            while True:
                try:
                    # Read available data (up to 1024 bytes)
                    chunk = await stream.read(1024)
                    if not chunk:
                        if buffer:
                            line_str = buffer.decode('utf-8', errors='ignore').strip()
                            if line_str:
                                line_count += 1
                                timestamp = time.strftime("%H:%M:%S")
                                print(f"[{timestamp}] [{prefix}] Line {line_count}: {line_str}")
                                yield line_str
                        print(f"[{prefix}] Stream ended after {line_count} lines")
                        break

                    buffer += chunk

                    # Process complete lines (split on both \r and \n)
                    # Replace \r\n with \n, then \r with \n for consistent splitting
                    buffer_str = buffer.decode('utf-8', errors='ignore')
                    buffer_str = buffer_str.replace('\r\n', '\n').replace('\r', '\n')
                    lines = buffer_str.split('\n')

                    # Keep last incomplete line in buffer
                    buffer = lines[-1].encode('utf-8')

                    # Process complete lines
                    for line in lines[:-1]:
                        line_str = line.strip()
                        if line_str:  # Only process non-empty lines
                            line_count += 1
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"[{timestamp}] [{prefix}] Line {line_count}: {line_str}")
                            yield line_str

                except Exception as e:
                    print(f"[{prefix}] Error reading stream: {e}")
                    break

        async def monitor_stdout():
            """Monitor stdout and parse progress"""
            total_frames = 240
            current_frame = 0

            async for line_str in read_stream(process.stdout, "gen_videos.py"):
                if not websocket:
                    continue

                # Try to parse tqdm progress
                # Pattern 1: "X/Y" format (e.g., "10/240")
                match = re.search(r'(\d+)/(\d+)', line_str)
                if match:
                    current_frame = int(match.group(1))
                    total_frames = int(match.group(2))
                    progress = current_frame / total_frames

                    await websocket.send_json({
                        "type": "progress",
                        "stage": "synthesis",
                        "progress": progress,
                        "step": current_frame,
                        "total_steps": total_frames,
                        "message": f"Generating frame {current_frame}/{total_frames}"
                    })
                    continue

                # Pattern 2: Percentage at start of line
                match = re.search(r'^\s*(\d+)%', line_str)
                if match:
                    percent = int(match.group(1))
                    progress = percent / 100.0

                    await websocket.send_json({
                        "type": "progress",
                        "stage": "synthesis",
                        "progress": progress,
                        "step": int(percent),
                        "total_steps": 100,
                        "message": f"Processing... {percent}%"
                    })
                    continue

                # Status messages
                if "Loading networks" in line_str:
                    await websocket.send_json({
                        "type": "progress",
                        "stage": "synthesis",
                        "progress": 0.05,
                        "message": "Loading SphereHead model..."
                    })
                elif "Starting video generation" in line_str:
                    await websocket.send_json({
                        "type": "progress",
                        "stage": "synthesis",
                        "progress": 0.1,
                        "message": "Starting video generation..."
                    })

        async def monitor_stderr():
            """Monitor stderr for tqdm progress (tqdm writes to stderr!)"""
            total_frames = 240
            current_frame = 0

            async for line_str in read_stream(process.stderr, "gen_videos.py STDERR"):
                if not websocket:
                    continue

                # Parse tqdm progress from stderr (where tqdm actually outputs!)
                # Pattern: "X/Y" format (e.g., "10/240")
                match = re.search(r'(\d+)/(\d+)', line_str)
                if match:
                    current_frame = int(match.group(1))
                    total_frames = int(match.group(2))
                    progress = current_frame / total_frames

                    await websocket.send_json({
                        "type": "progress",
                        "stage": "synthesis",
                        "progress": progress,
                        "step": current_frame,
                        "total_steps": total_frames,
                        "message": f"Generating frame {current_frame}/{total_frames}"
                    })
                    print(f"[Progress] Sent update: {current_frame}/{total_frames} ({int(progress*100)}%)")
                    continue

                # Check for errors
                if "error" in line_str.lower() or "exception" in line_str.lower():
                    await websocket.send_json({
                        "type": "progress",
                        "stage": "synthesis",
                        "progress": 0.0,
                        "message": f"Warning: {line_str[:100]}"
                    })

        # Run both monitors concurrently
        await asyncio.gather(
            monitor_stdout(),
            monitor_stderr(),
            return_exceptions=True
        )

    async def _organize_outputs(
        self,
        output_dir: Path,
        session_id: str,
        seed: int,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find generated files and organize them with standard names

        gen_videos.py outputs:
        - {outdir}/checkpoint-025000_90.mp4 (main video)
        - {outdir}/checkpoint-025000_90_depth.mp4 (depth video, if generated)
        """
        # Find the generated video file
        # Pattern: checkpoint-025000_90.mp4 (model name + pose angle)
        video_files = list(output_dir.glob("*.mp4"))

        if not video_files:
            raise RuntimeError(f"No video file found in {output_dir}")

        # Find main video (not depth)
        main_video = None
        depth_video = None

        for vf in video_files:
            if "_depth" in vf.name:
                depth_video = vf
            else:
                main_video = vf

        if not main_video:
            raise RuntimeError(f"No main video file found in {output_dir}")

        # Rename to standard names
        video_path = output_dir / "video.mp4"
        if main_video != video_path:
            main_video.rename(video_path)

        result = {
            "imageUrl": f"/api/outputs/{session_id}/video.mp4",  # Using video as preview
            "latentCodePath": f"/api/outputs/{session_id}/latent.npz",  # Placeholder
            "metadata": {
                "seed": seed,
                "method": "seed",
                "params": metadata
            }
        }

        if depth_video:
            depth_path = output_dir / "depth.mp4"
            if depth_video != depth_path:
                depth_video.rename(depth_path)
            result["depthUrl"] = f"/api/outputs/{session_id}/depth.mp4"

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return result


# Global service instance
spherehead_service = SphereHeadService()
