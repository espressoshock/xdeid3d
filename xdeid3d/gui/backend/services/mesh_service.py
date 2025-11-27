"""
Mesh Generation Service
Handles 3D mesh extraction from generated identities using SphereHead
"""
import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, Callable
import traceback
import trimesh
import numpy as np

# Add core directory to path
CORE_DIR = Path(__file__).parent.parent.parent.parent / "core"
sys.path.insert(0, str(CORE_DIR))


class MeshService:
    """
    Service for generating 3D meshes from identities

    Uses gen_videos.py with --shapes flag to extract .ply mesh files
    """

    def __init__(self, model_path: str):
        """
        Initialize mesh service

        Args:
            model_path: Path to SphereHead model checkpoint (.pkl file)
        """
        self.model_path = model_path
        self.core_dir = CORE_DIR

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    async def generate_mesh(
        self,
        session_id: str,
        output_dir: Path,
        latent_code_path: Optional[str] = None,
        generator_path: Optional[str] = None,
        seed: Optional[int] = None,
        truncation: float = 0.65,
        voxel_res: int = 512,
        force_regenerate: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Generate 3D mesh from identity

        Args:
            session_id: Unique session identifier
            output_dir: Directory to save mesh file
            latent_code_path: Path to .npz latent code (for PTI-based identities)
            generator_path: Path to fine-tuned generator (for PTI)
            seed: Random seed (for seed-based identities)
            truncation: Truncation psi value
            voxel_res: Voxel resolution (512 or 1024)
            force_regenerate: Force regeneration even if cached mesh exists
            progress_callback: Optional callback for progress updates (progress, message)

        Returns:
            Path to generated .ply mesh file

        Raises:
            ValueError: If neither latent_code_path nor seed is provided
            RuntimeError: If mesh generation fails
        """
        # Validate input
        if not latent_code_path and seed is None:
            raise ValueError("Must provide either latent_code_path or seed")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if mesh already exists (cache check)
        # Cache filename includes voxel resolution to avoid conflicts
        if seed is not None:
            expected_mesh = output_dir / f"seed{seed:04d}_voxel{voxel_res}.ply"
            if expected_mesh.exists() and not force_regenerate:
                print(f"[MeshService] Mesh already exists, using cached version: {expected_mesh}")
                if progress_callback:
                    result = progress_callback(1.0, "Using cached mesh")
                    if asyncio.iscoroutine(result):
                        await result
                return str(expected_mesh)
            elif expected_mesh.exists() and force_regenerate:
                print(f"[MeshService] Force regenerate enabled, deleting cached mesh: {expected_mesh}")
                expected_mesh.unlink()  # Delete cached mesh

        # Create progress file for real-time monitoring
        progress_file = output_dir / f"mesh_progress_{session_id}.json"

        # Delete old progress file to prevent reading stale data
        if progress_file.exists():
            progress_file.unlink()
            print(f"[MeshService] Deleted old progress file: {progress_file}")

        # Build command
        cmd = self._build_command(
            latent_code_path=latent_code_path,
            generator_path=generator_path,
            seed=seed,
            truncation=truncation,
            voxel_res=voxel_res,
            output_dir=output_dir,
            progress_file=progress_file,
        )

        print(f"[MeshService] Running command: {' '.join(cmd)}")
        print(f"[MeshService] Working directory: {self.core_dir}")

        # Report initial progress
        if progress_callback:
            result = progress_callback(0.0, "Initializing mesh generation...")
            if asyncio.iscoroutine(result):
                await result

        # Execute subprocess
        try:
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
            print(f"[MeshService] Streaming output from gen_videos.py:")
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

            # Check return code
            if process.returncode != 0:
                error_msg = '\n'.join(output_lines[-10:]) if output_lines else "Unknown error"
                print(f"[MeshService] ERROR: Process failed with code {process.returncode}")
                print(f"[MeshService] Last 10 lines: {error_msg}")
                raise RuntimeError(f"Mesh generation failed: {error_msg}")

            print(f"[MeshService] Process completed successfully")

            # Find generated mesh file
            mesh_path = self._find_mesh_file(output_dir, seed, voxel_res)

            if not mesh_path or not os.path.exists(mesh_path):
                raise RuntimeError(f"Mesh file not found in {output_dir}")

            # Final progress update
            if progress_callback:
                result = progress_callback(1.0, "Mesh generation complete")
                if asyncio.iscoroutine(result):
                    await result

            # Clean up progress file
            if progress_file.exists():
                progress_file.unlink()

            return str(mesh_path)

        except Exception as e:
            print(f"[MeshService] Exception during mesh generation:")
            traceback.print_exc()

            if progress_callback:
                progress_callback(0.0, f"Error: {str(e)}")

            # Clean up progress file
            if progress_file.exists():
                progress_file.unlink()

            raise RuntimeError(f"Mesh generation failed: {str(e)}")

    def _build_command(
        self,
        latent_code_path: Optional[str],
        generator_path: Optional[str],
        seed: Optional[int],
        truncation: float,
        voxel_res: int,
        output_dir: Path,
        progress_file: Path,
    ) -> list:
        """Build gen_videos.py command with appropriate parameters"""

        # Use the xdeid3d conda environment's Python explicitly
        # This ensures all dependencies (plyfile, scikit-image, PyMCubes) are available
        python_path = '/home/ubuntu/miniconda3/envs/xdeid3d/bin/python'

        cmd = [
            python_path, 'gen_videos.py',
            '--network', self.model_path,
            '--shapes', 'true',  # Enable mesh extraction
            '--shapes_only', 'true',  # Skip video generation, only extract mesh
            '--voxel_res', str(voxel_res),
            '--trunc', str(truncation),
            '--outdir', str(output_dir),
            '--cfg', 'Head',
            '--w-frames', '1',  # Generate only 1 frame (not 240) - we just need final mesh
            '--progress-file', str(progress_file),
        ]

        # Add seed or latent code path
        if seed is not None:
            cmd.extend(['--seeds', str(seed)])
        elif latent_code_path:
            # For PTI-based identities, we need to generate from latent code
            # gen_videos.py doesn't directly support loading .npz files
            # We'll use a seed of 0 and then manually load the latent code
            # This is a limitation - in production, we'd need to modify gen_videos.py
            # For now, fall back to seed-based generation
            print(f"[MeshService] WARNING: PTI-based mesh generation not fully supported yet")
            print(f"[MeshService] Using seed=0 as fallback (mesh may not match identity)")
            cmd.extend(['--seeds', '0'])

        return cmd

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
                        message = f"{desc} ({current}/{total})" if desc else f"Generating mesh... {int(progress * 100)}%"
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

    def _clean_mesh(self, mesh_path: Path) -> bool:
        """
        Clean mesh by removing floating islands (disconnected components)

        Keeps only the largest connected component, which should be the main head mesh.
        Removes all smaller disconnected artifacts.

        Args:
            mesh_path: Path to the mesh file to clean (will be modified in-place)

        Returns:
            True if mesh was cleaned successfully, False otherwise
        """
        try:
            print(f"[MeshService] Loading mesh for cleaning: {mesh_path}")

            # Load mesh using trimesh
            mesh = trimesh.load(str(mesh_path))

            # Get connected components (split mesh into separate pieces)
            components = mesh.split(only_watertight=False)

            if len(components) <= 1:
                print(f"[MeshService] Mesh already clean (single component)")
                return True

            print(f"[MeshService] Found {len(components)} connected components")

            # Find the largest component by number of faces
            largest_component = max(components, key=lambda c: len(c.faces))

            # Count vertices/faces in largest vs removed
            total_vertices = sum(len(c.vertices) for c in components)
            total_faces = sum(len(c.faces) for c in components)
            kept_vertices = len(largest_component.vertices)
            kept_faces = len(largest_component.faces)
            removed_vertices = total_vertices - kept_vertices
            removed_faces = total_faces - kept_faces

            print(f"[MeshService] Kept largest component:")
            print(f"  - Vertices: {kept_vertices}/{total_vertices} ({kept_vertices/total_vertices*100:.1f}%)")
            print(f"  - Faces: {kept_faces}/{total_faces} ({kept_faces/total_faces*100:.1f}%)")
            print(f"[MeshService] Removed {len(components)-1} floating islands:")
            print(f"  - Vertices: {removed_vertices}")
            print(f"  - Faces: {removed_faces}")

            # Export cleaned mesh (overwrite original)
            largest_component.export(str(mesh_path))

            print(f"[MeshService] Successfully cleaned mesh: {mesh_path}")
            return True

        except Exception as e:
            print(f"[MeshService] Error cleaning mesh: {e}")
            traceback.print_exc()
            # Return True to continue even if cleaning fails
            # Better to have uncleaned mesh than no mesh at all
            return True

    def _find_mesh_file(self, output_dir: Path, seed: Optional[int], voxel_res: int) -> Optional[str]:
        """
        Find generated mesh file and move it to output directory

        gen_videos.py generates meshes in core/shape_seed_{seed}/ directory
        with names like 0000_shape.ply, 0001_shape.ply, etc.

        Args:
            output_dir: Directory where mesh should be copied
            seed: Seed used for generation
            voxel_res: Voxel resolution (included in filename)

        Returns:
            Path to mesh file in output_dir
        """
        if seed is None:
            seed = 0

        # gen_videos.py writes shapes to shape_seed_{seed}/ in the working directory
        shape_dir = self.core_dir / f"shape_seed_{seed}"

        print(f"[MeshService] Looking for mesh in: {shape_dir}")

        if not shape_dir.exists():
            print(f"[MeshService] Shape directory not found: {shape_dir}")
            return None

        # Find .ply files (with --w-frames 1, should be just one: 0000_shape.ply)
        ply_files = sorted(shape_dir.glob("*_shape.ply"))

        if not ply_files:
            print(f"[MeshService] No .ply files found in {shape_dir}")
            return None

        # Use the last generated mesh file (or only file if --w-frames 1)
        source_mesh = ply_files[-1]
        print(f"[MeshService] Found mesh file: {source_mesh} ({len(ply_files)} total frames)")

        # Copy mesh to output directory with clean name (including voxel resolution)
        import shutil
        dest_mesh = output_dir / f"seed{seed:04d}_voxel{voxel_res}.ply"
        shutil.copy2(source_mesh, dest_mesh)
        print(f"[MeshService] Copied mesh to: {dest_mesh}")

        # Clean mesh to remove floating islands
        print(f"[MeshService] Cleaning mesh to remove floating islands...")
        self._clean_mesh(dest_mesh)

        # Clean up shape directory
        try:
            shutil.rmtree(shape_dir)
            print(f"[MeshService] Cleaned up shape directory: {shape_dir}")
        except Exception as e:
            print(f"[MeshService] Warning: Could not clean up shape directory: {e}")

        return str(dest_mesh)


# Singleton instance
_mesh_service: Optional[MeshService] = None


def get_mesh_service(model_path: str) -> MeshService:
    """
    Get or create singleton mesh service instance

    Args:
        model_path: Path to SphereHead model checkpoint

    Returns:
        MeshService instance
    """
    global _mesh_service

    if _mesh_service is None:
        _mesh_service = MeshService(model_path)
        print(f"[MeshService] Initialized with model: {model_path}")

    return _mesh_service
