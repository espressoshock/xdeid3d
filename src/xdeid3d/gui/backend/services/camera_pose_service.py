"""
Camera Pose Video Generation Service
Handles video synthesis with custom camera trajectories
"""
import os
import sys
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Callable, Tuple
import traceback

# Add core directory to path
CORE_DIR = Path(__file__).parent.parent.parent.parent / "core"
sys.path.insert(0, str(CORE_DIR))


class CameraPose:
    """Camera pose with spherical coordinates"""
    def __init__(self, yaw: float, pitch: float, radius: float):
        self.yaw = yaw      # Horizontal angle (radians)
        self.pitch = pitch  # Vertical angle from Y axis (radians)
        self.radius = radius  # Distance from origin


class CameraPoseService:
    """
    Service for generating videos with custom camera trajectories

    Uses gen_videos.py with modified camera sampling to create videos
    with specific camera paths
    """

    def __init__(self, model_path: str):
        """
        Initialize camera pose service

        Args:
            model_path: Path to SphereHead model checkpoint (.pkl file)
        """
        self.model_path = model_path
        self.core_dir = CORE_DIR

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    def get_preset_trajectory(
        self,
        preset_id: str,
        num_frames: int,
        duration: float,
    ) -> List[CameraPose]:
        """
        Generate camera trajectory from preset

        Args:
            preset_id: Preset identifier
            num_frames: Number of frames to generate
            duration: Total duration in seconds

        Returns:
            List of camera poses
        """
        if preset_id == 'rotate360':
            # Full 360° horizontal rotation
            return [
                CameraPose(
                    yaw=2 * np.pi * i / num_frames,
                    pitch=np.pi / 2,  # Level with horizon
                    radius=2.7
                )
                for i in range(num_frames)
            ]

        elif preset_id == 'orbit':
            # Circular orbit with pitch variation
            return [
                CameraPose(
                    yaw=2 * np.pi * i / num_frames,
                    pitch=np.pi / 2 + 0.3 * np.sin(2 * np.pi * i / num_frames),
                    radius=2.7
                )
                for i in range(num_frames)
            ]

        elif preset_id == 'sidebyside':
            # Left-right oscillation
            yaw_range = np.pi / 3  # 60 degrees total range
            return [
                CameraPose(
                    yaw=np.pi / 2 + yaw_range * np.sin(2 * np.pi * i / num_frames),
                    pitch=np.pi / 2,
                    radius=2.7
                )
                for i in range(num_frames)
            ]

        elif preset_id == 'front':
            # Static frontal view
            return [
                CameraPose(
                    yaw=np.pi / 2,
                    pitch=np.pi / 2,
                    radius=2.7
                )
                for i in range(num_frames)
            ]

        else:
            raise ValueError(f"Unknown preset: {preset_id}")

    def interpolate_keyframes(
        self,
        keyframes: List[dict],
        num_frames: int,
        interpolation: str = 'cubic',
    ) -> List[CameraPose]:
        """
        Interpolate camera trajectory from keyframes

        Args:
            keyframes: List of keyframes with timestamp (0-1), yaw, pitch, radius
            num_frames: Number of frames to generate
            interpolation: 'linear' or 'cubic'

        Returns:
            List of camera poses
        """
        if len(keyframes) < 2:
            raise ValueError("Need at least 2 keyframes for interpolation")

        # Sort keyframes by timestamp
        keyframes = sorted(keyframes, key=lambda k: k['timestamp'])

        # Extract timestamps and values
        timestamps = np.array([kf['timestamp'] for kf in keyframes])
        yaws = np.array([kf['yaw'] for kf in keyframes])
        pitches = np.array([kf['pitch'] for kf in keyframes])
        radii = np.array([kf['radius'] for kf in keyframes])

        # Normalize timestamps to frame indices
        frame_indices = timestamps * (num_frames - 1)

        # Create interpolation points
        t = np.linspace(0, num_frames - 1, num_frames)

        if interpolation == 'cubic' and len(keyframes) >= 4:
            # Use cubic spline interpolation
            from scipy import interpolate
            yaw_interp = interpolate.CubicSpline(frame_indices, yaws, bc_type='natural')
            pitch_interp = interpolate.CubicSpline(frame_indices, pitches, bc_type='natural')
            radius_interp = interpolate.CubicSpline(frame_indices, radii, bc_type='natural')
        else:
            # Use linear interpolation
            yaw_interp = lambda x: np.interp(x, frame_indices, yaws)
            pitch_interp = lambda x: np.interp(x, frame_indices, pitches)
            radius_interp = lambda x: np.interp(x, frame_indices, radii)

        # Generate trajectory
        trajectory = []
        for i in range(num_frames):
            trajectory.append(CameraPose(
                yaw=float(yaw_interp(t[i])),
                pitch=float(pitch_interp(t[i])),
                radius=float(radius_interp(t[i])),
            ))

        return trajectory

    async def render_preview(
        self,
        session_id: str,
        output_dir: Path,
        trajectory: List[CameraPose],
        latent_code_path: Optional[str] = None,
        generator_path: Optional[str] = None,
        seed: Optional[int] = None,
        truncation: float = 0.65,
        num_frames: int = 60,
        fps: int = 24,
        quality: str = 'medium',
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Render preview video with custom camera trajectory (optimized for quick preview)

        Args:
            session_id: Unique session identifier
            output_dir: Directory to save video file
            trajectory: List of camera poses
            latent_code_path: Path to .npz latent code (for PTI)
            generator_path: Path to fine-tuned generator (for PTI)
            seed: Random seed (for seed-based identities)
            truncation: Truncation psi value
            num_frames: Number of frames for preview (30, 60, 120, 240)
            fps: Frames per second (24 or 30)
            quality: Preview quality ('low', 'medium', 'high')
            progress_callback: Optional callback for progress updates

        Returns:
            Path to generated preview video file

        Raises:
            ValueError: If neither latent_code_path nor seed is provided
            RuntimeError: If video generation fails
        """
        # Validate input
        if not latent_code_path and seed is None:
            raise ValueError("Must provide either latent_code_path or seed")

        if len(trajectory) != num_frames:
            print(f"[CameraPoseService] Warning: trajectory length ({len(trajectory)}) != num_frames ({num_frames})")

        # Ensure output directory exists
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        # Save trajectory to JSON file for gen_videos.py to read
        trajectory_file = preview_dir / f"preview_trajectory_{session_id}.json"
        self._save_trajectory(trajectory, trajectory_file)

        # Map quality to rendering parameters
        quality_settings = {
            'low': {'nrr': 64, 'sample_mult': 1.0},
            'medium': {'nrr': 128, 'sample_mult': 1.5},
            'high': {'nrr': 256, 'sample_mult': 2.0},
            'best': {'nrr': 512, 'sample_mult': 3.0},
        }

        if quality not in quality_settings:
            print(f"[CameraPoseService] Unknown quality '{quality}', using 'medium'")
            quality = 'medium'

        settings = quality_settings[quality]

        # Progress file for monitoring (must be created before command)
        progress_file = preview_dir / f"preview_progress_{session_id}.json"

        # Clean up old video files to prevent caching issues
        # Delete any existing .mp4 files in the preview directory
        for old_video in preview_dir.glob("*.mp4"):
            try:
                old_video.unlink()
                print(f"[CameraPoseService] Deleted old video file: {old_video.name}")
            except Exception as e:
                print(f"[CameraPoseService] Warning: Could not delete {old_video.name}: {e}")

        # Build command
        cmd = self._build_preview_command(
            latent_code_path=latent_code_path,
            generator_path=generator_path,
            seed=seed,
            truncation=truncation,
            num_frames=num_frames,
            fps=fps,
            nrr=settings['nrr'],
            sample_mult=settings['sample_mult'],
            output_dir=preview_dir,
            trajectory_file=trajectory_file,
            session_id=session_id,
            progress_file=progress_file,
        )

        print(f"[CameraPoseService] Rendering preview with quality={quality} (nrr={settings['nrr']}, sample_mult={settings['sample_mult']})")
        print(f"[CameraPoseService] Command: {' '.join(cmd)}")

        # Report initial progress
        if progress_callback:
            result = progress_callback(0.0, "Initializing preview rendering...")
            if asyncio.iscoroutine(result):
                await result

        # Progress file for monitoring
        progress_file = preview_dir / f"preview_progress_{session_id}.json"

        # Delete old progress file to prevent reading stale data
        if progress_file.exists():
            progress_file.unlink()
            print(f"[CameraPoseService] Deleted old progress file: {progress_file}")

        try:
            # Run subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.core_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
            )

            # Monitor progress file in background
            monitor_task = asyncio.create_task(
                self._monitor_progress(progress_file, progress_callback)
            )

            # Stream output in real-time
            print(f"[CameraPoseService] Streaming output from gen_videos.py:")
            output_lines = []
            async for line in process.stdout:
                line_str = line.decode().strip()
                if line_str:
                    print(f"[gen_videos.py] {line_str}")
                    output_lines.append(line_str)

            # Wait for process to complete
            await process.wait()

            # Stop progress monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            if process.returncode != 0:
                error_msg = "\n".join(output_lines[-10:]) if output_lines else "Unknown error"
                print(f"[CameraPoseService] Preview rendering failed: {error_msg}")
                raise RuntimeError(f"Preview rendering failed: {error_msg}")

            print(f"[CameraPoseService] Preview rendering completed successfully")

            # Find generated video file
            video_path = preview_dir / f"preview_{session_id}.mp4"

            # gen_videos.py creates files with network name + pose_cond naming
            # e.g., "spherehead-ckpt-025000_90.mp4"
            network_basename = os.path.splitext(os.path.basename(self.model_path))[0]
            generated_video = preview_dir / f"{network_basename}_90.mp4"

            print(f"[CameraPoseService] Looking for generated video: {generated_video}")

            if generated_video.exists():
                file_size = generated_video.stat().st_size
                print(f"[CameraPoseService] Found generated video: {generated_video.name} ({file_size} bytes)")

                if file_size < 1000:
                    print(f"[CameraPoseService] WARNING: Video file is suspiciously small ({file_size} bytes)")
                    print(f"[CameraPoseService] This might indicate a rendering error")

                # Remove destination if it exists (ensure clean overwrite)
                if video_path.exists():
                    video_path.unlink()
                    print(f"[CameraPoseService] Removed old preview file")

                # Rename to our expected name
                generated_video.rename(video_path)
                print(f"[CameraPoseService] Renamed {generated_video.name} -> {video_path.name}")
            else:
                # Try to find any .mp4 file as fallback
                mp4_files = list(preview_dir.glob("*.mp4"))
                print(f"[CameraPoseService] Generated video not found at expected path")
                print(f"[CameraPoseService] Found {len(mp4_files)} .mp4 files in directory")

                if mp4_files:
                    for f in mp4_files:
                        print(f"[CameraPoseService]   - {f.name} ({f.stat().st_size} bytes)")

                    # Use the largest file (most likely to be the actual video)
                    largest_file = max(mp4_files, key=lambda f: f.stat().st_size)
                    print(f"[CameraPoseService] Using largest file: {largest_file.name}")

                    # Remove destination if it exists
                    if video_path.exists():
                        video_path.unlink()

                    largest_file.rename(video_path)
                    print(f"[CameraPoseService] Renamed {largest_file.name} -> {video_path.name}")
                else:
                    raise RuntimeError(f"No video file found in {preview_dir}")

            if progress_callback:
                result = progress_callback(1.0, "Preview rendering complete")
                if asyncio.iscoroutine(result):
                    await result

            return str(video_path)

        except Exception as e:
            print(f"[CameraPoseService] Preview rendering error: {str(e)}")
            print(f"[CameraPoseService] Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Preview rendering failed: {str(e)}")

    async def generate_pose_video(
        self,
        session_id: str,
        output_dir: Path,
        trajectory: List[CameraPose],
        latent_code_path: Optional[str] = None,
        generator_path: Optional[str] = None,
        seed: Optional[int] = None,
        truncation: float = 0.65,
        num_frames: int = 240,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Generate video with custom camera trajectory

        Args:
            session_id: Unique session identifier
            output_dir: Directory to save video file
            trajectory: List of camera poses
            latent_code_path: Path to .npz latent code (for PTI)
            generator_path: Path to fine-tuned generator (for PTI)
            seed: Random seed (for seed-based identities)
            truncation: Truncation psi value
            num_frames: Number of frames (should match trajectory length)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to generated video file

        Raises:
            ValueError: If neither latent_code_path nor seed is provided
            RuntimeError: If video generation fails
        """
        # Validate input
        if not latent_code_path and seed is None:
            raise ValueError("Must provide either latent_code_path or seed")

        if len(trajectory) != num_frames:
            print(f"[CameraPoseService] Warning: trajectory length ({len(trajectory)}) != num_frames ({num_frames})")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save trajectory to JSON file for gen_videos.py to read
        trajectory_file = output_dir / f"camera_trajectory_{session_id}.json"
        self._save_trajectory(trajectory, trajectory_file)

        # Build command
        cmd = self._build_command(
            latent_code_path=latent_code_path,
            generator_path=generator_path,
            seed=seed,
            truncation=truncation,
            num_frames=num_frames,
            output_dir=output_dir,
            trajectory_file=trajectory_file,
        )

        print(f"[CameraPoseService] Running command: {' '.join(cmd)}")

        # Report initial progress
        if progress_callback:
            progress_callback(0.0, "Initializing video generation...")

        # For now, return placeholder since gen_videos.py doesn't support custom trajectories yet
        # In production, we'd need to modify gen_videos.py to accept --camera-trajectory flag

        # Create placeholder video path
        video_path = output_dir / f"pose_video_{session_id}.mp4"

        # TODO: Actually run subprocess when gen_videos.py supports custom trajectories
        print(f"[CameraPoseService] TODO: Implement actual video generation")
        print(f"[CameraPoseService] Trajectory saved to: {trajectory_file}")
        print(f"[CameraPoseService] Would generate video at: {video_path}")

        if progress_callback:
            progress_callback(1.0, "Video generation complete (placeholder)")

        # For now, return the trajectory file path as a placeholder
        # In production, this would be the actual video file
        return str(trajectory_file)

    async def _monitor_progress(
        self,
        progress_file: Path,
        callback: Optional[Callable[[float, str], None]],
        poll_interval: float = 0.5,
    ):
        """
        Monitor progress file and report updates

        Args:
            progress_file: Path to JSON progress file
            callback: Progress callback function
            poll_interval: How often to check file (seconds)
        """
        if not callback:
            return

        last_progress = 0.0

        try:
            while True:
                await asyncio.sleep(poll_interval)

                if not progress_file.exists():
                    continue

                try:
                    with open(progress_file, 'r') as f:
                        data = json.load(f)

                    progress = data.get('progress', 0.0)
                    desc = data.get('desc', '')
                    current = data.get('current', 0)
                    total = data.get('total', 100)

                    # Clamp progress to [0.0, 1.0] to prevent >100% display
                    progress = max(0.0, min(1.0, progress))

                    # Only report if progress changed
                    if abs(progress - last_progress) > 0.01:
                        message = f"{desc} ({current}/{total})" if desc else f"Rendering preview... {int(progress * 100)}%"
                        # Call callback (may be async)
                        result = callback(progress, message)
                        if asyncio.iscoroutine(result):
                            await result
                        last_progress = progress

                except (json.JSONDecodeError, IOError):
                    # File might be being written, ignore
                    pass

        except asyncio.CancelledError:
            # Task cancelled, exit gracefully
            pass

    def _save_trajectory(self, trajectory: List[CameraPose], filepath: Path):
        """Save camera trajectory to JSON file"""
        data = {
            'num_frames': len(trajectory),
            'poses': [
                {
                    'yaw': pose.yaw,
                    'pitch': pose.pitch,
                    'radius': pose.radius,
                }
                for pose in trajectory
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[CameraPoseService] Saved trajectory with {len(trajectory)} frames to {filepath}")

    def _build_preview_command(
        self,
        latent_code_path: Optional[str],
        generator_path: Optional[str],
        seed: Optional[int],
        truncation: float,
        num_frames: int,
        fps: int,
        nrr: int,
        sample_mult: float,
        output_dir: Path,
        trajectory_file: Path,
        session_id: str,
        progress_file: Path,
    ) -> list:
        """Build command for preview rendering with quality settings"""

        # Use the xdeid3d conda environment's Python explicitly
        python_path = '/home/ubuntu/miniconda3/envs/xdeid3d/bin/python'

        # Choose script and parameters based on identity type
        if latent_code_path and generator_path:
            # PTI/Projected identity - use specialized script
            print(f"[CameraPoseService] Using PTI identity with fine-tuned generator")
            print(f"[CameraPoseService]   Generator: {generator_path}")
            print(f"[CameraPoseService]   Latent code: {latent_code_path}")

            cmd = [
                python_path, 'gen_videos_proj_withseg.py',  # Use PTI script
                '--network', generator_path,                 # Fine-tuned generator
                '--projected-w', latent_code_path,           # Projected latent code
                '--trunc', str(truncation),
                '--outdir', str(output_dir),
                '--cfg', 'Head',
                '--w-frames', str(num_frames),
                '--nrr', str(nrr),
                '--sample_mult', str(sample_mult),
                '--progress-file', str(progress_file),
                '--camera-trajectory', str(trajectory_file),
            ]
        else:
            # Seed-based identity - use standard script
            print(f"[CameraPoseService] Using seed-based identity: {seed}")

            cmd = [
                python_path, 'gen_videos.py',               # Use seed script
                '--network', self.model_path,               # Base model
                '--trunc', str(truncation),
                '--outdir', str(output_dir),
                '--cfg', 'Head',
                '--w-frames', str(num_frames),
                '--nrr', str(nrr),
                '--sample_mult', str(sample_mult),
                '--progress-file', str(progress_file),
                '--camera-trajectory', str(trajectory_file),
            ]

            # Add seed
            if seed is not None:
                cmd.extend(['--seeds', str(seed)])
            else:
                # Default seed
                cmd.extend(['--seeds', '0'])

        return cmd

    def _build_command(
        self,
        latent_code_path: Optional[str],
        generator_path: Optional[str],
        seed: Optional[int],
        truncation: float,
        num_frames: int,
        output_dir: Path,
        trajectory_file: Path,
    ) -> list:
        """Build gen_videos.py command with camera trajectory"""

        # Use the xdeid3d conda environment's Python explicitly
        # This ensures all dependencies are available
        python_path = '/home/ubuntu/miniconda3/envs/xdeid3d/bin/python'

        cmd = [
            python_path, 'gen_videos.py',
            '--network', self.model_path,
            '--trunc', str(truncation),
            '--outdir', str(output_dir),
            '--cfg', 'Head',
            '--w-frames', str(num_frames),
            # TODO: Add --camera-trajectory flag when implemented
            # '--camera-trajectory', str(trajectory_file),
        ]

        # Add seed
        if seed is not None:
            cmd.extend(['--seeds', str(seed)])
        else:
            # For PTI, use seed 0 as fallback
            cmd.extend(['--seeds', '0'])

        return cmd


# Singleton instance
_camera_pose_service: Optional[CameraPoseService] = None


def get_camera_pose_service(model_path: str) -> CameraPoseService:
    """
    Get or create singleton camera pose service instance

    Args:
        model_path: Path to SphereHead model checkpoint

    Returns:
        CameraPoseService instance
    """
    global _camera_pose_service

    if _camera_pose_service is None:
        _camera_pose_service = CameraPoseService(model_path)
        print(f"[CameraPoseService] Initialized with model: {model_path}")

    return _camera_pose_service
