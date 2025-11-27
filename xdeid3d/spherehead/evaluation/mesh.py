#!/usr/bin/env python3
"""
SphereHead-GUARD 3D Mesh Evaluation Script
Creates visualization with 3D heatmap meshes colored by each metric's performance
"""

import os
import sys
import json
import subprocess
import argparse
import torch
import numpy as np
import imageio
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Package imports (no sys.path manipulation needed)

from xdeid3d.spherehead import dnnlib
from xdeid3d.spherehead.camera_utils import LookAtPoseSampler
from xdeid3d.spherehead.torch_utils import misc
from xdeid3d.spherehead.training.triplane import TriPlaneGenerator
from xdeid3d.spherehead.evaluation.gpu_utils import get_best_gpu_device, clear_gpu_memory

# Import functions from evaluate_synthetic.py
from xdeid3d.spherehead.evaluation.synthetic import create_samples, load_network_pkl_cpu_safe

# Check for optional imports
try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available, mesh rendering will be limited")

try:
    import pyrender

    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class MetricMeshRenderer:
    """Renders 3D meshes colored by metric values"""

    def __init__(
        self, metric_name: str, device: torch.device, vertex_batch_size: int = 128
    ):
        self.metric_name = metric_name
        self.device = device
        self.mesh_path = None
        self.metric_values_by_angle = {}  # Store metric values for each viewing angle
        self.vertex_batch_size = vertex_batch_size

    def add_metric_value(self, yaw: float, pitch: float, value: float):
        """Store metric value for a specific viewing angle"""
        # Quantize angles to reduce memory usage
        yaw_deg = int(np.degrees(yaw) % 360)
        pitch_deg = int(np.degrees(pitch))
        self.metric_values_by_angle[(yaw_deg, pitch_deg)] = value

    def create_metric_colored_mesh(
        self,
        sigmas: np.ndarray,
        voxel_origin: np.ndarray,
        voxel_size: float,
        output_path: str,
        colormap: str = "magma",
    ) -> str:
        """
        Create a mesh colored by metric values at different viewing angles.

        This is similar to create_colored_mesh_ply but uses actual metric values
        instead of fake neural network scores.
        """
        import skimage.measure
        from plyfile import PlyData, PlyElement
        from scipy import ndimage

        print(f"Generating colored mesh for metric: {self.metric_name}")

        # Pre-process sigmas
        sigmas_smooth = ndimage.gaussian_filter(sigmas, sigma=0.5)
        pad_width = 2
        sigmas_padded = np.pad(
            sigmas_smooth, pad_width, mode="constant", constant_values=0
        )

        # Extract mesh
        level = 10.0
        try:
            verts, faces, normals, values = skimage.measure.marching_cubes(
                sigmas_padded,
                level=level,
                spacing=[voxel_size] * 3,
                step_size=1,
                allow_degenerate=False,
            )
            verts -= pad_width * voxel_size
        except ValueError:
            verts, faces, normals, values = skimage.measure.marching_cubes(
                sigmas_padded, level=level, spacing=[voxel_size] * 3, step_size=2
            )
            verts -= pad_width * voxel_size

        # Transform vertices to world coordinates
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]

        # Color vertices based on metric values
        vertex_scores = np.zeros(len(mesh_points))

        print("Computing vertex colors based on metric values...")

        # Convert metric angles to arrays for vectorized computation
        metric_angles = np.array(list(self.metric_values_by_angle.keys()))
        metric_values = np.array(list(self.metric_values_by_angle.values()))

        # Process vertices in batches for better performance
        num_vertices = len(mesh_points)

        with tqdm(
            total=num_vertices, desc=f"Coloring vertices for {self.metric_name}"
        ) as pbar:
            for batch_start in range(0, num_vertices, self.vertex_batch_size):
                batch_end = min(batch_start + self.vertex_batch_size, num_vertices)
                batch_vertices = mesh_points[batch_start:batch_end]

                # Compute viewing angles for batch
                vertex_norms = (
                    np.linalg.norm(batch_vertices, axis=1, keepdims=True) + 1e-8
                )
                vertex_dirs = batch_vertices / vertex_norms

                # Convert to spherical coordinates
                vertex_yaws = np.arctan2(vertex_dirs[:, 2], vertex_dirs[:, 0]) + np.pi
                vertex_pitches = np.arccos(np.clip(vertex_dirs[:, 1], -1, 1))

                vertex_yaw_degs = np.degrees(vertex_yaws) % 360
                vertex_pitch_degs = np.degrees(vertex_pitches)

                # Compute scores for batch
                for i, (yaw_deg, pitch_deg) in enumerate(
                    zip(vertex_yaw_degs, vertex_pitch_degs)
                ):
                    if len(metric_angles) == 0:
                        vertex_scores[batch_start + i] = 0.5
                        continue

                    # Compute angular distances to all metric angles
                    yaw_dists = np.minimum(
                        np.abs(metric_angles[:, 0] - yaw_deg),
                        360 - np.abs(metric_angles[:, 0] - yaw_deg),
                    )
                    pitch_dists = np.abs(metric_angles[:, 1] - pitch_deg)
                    angle_dists = np.sqrt(yaw_dists**2 + pitch_dists**2)

                    # Gaussian weighting
                    weights = np.exp(-(angle_dists**2) / 500)
                    total_weight = np.sum(weights)

                    if total_weight > 0:
                        vertex_scores[batch_start + i] = (
                            np.sum(metric_values * weights) / total_weight
                        )
                    else:
                        vertex_scores[batch_start + i] = (
                            np.mean(metric_values) if len(metric_values) > 0 else 0.5
                        )

                pbar.update(batch_end - batch_start)

        # Normalize scores
        if vertex_scores.max() > vertex_scores.min():
            vertex_scores = (vertex_scores - vertex_scores.min()) / (
                vertex_scores.max() - vertex_scores.min()
            )

        # Convert to colors
        try:
            cmap = cm.colormaps[colormap]
        except AttributeError:
            cmap = cm.get_cmap(colormap)
        vertex_colors = cmap(vertex_scores)[:, :3]

        # Post-process mesh if trimesh is available
        if TRIMESH_AVAILABLE:
            print("Post-processing mesh...")
            temp_mesh = trimesh.Trimesh(vertices=mesh_points, faces=faces)

            # Basic cleanup
            temp_mesh.update_faces(temp_mesh.unique_faces())
            temp_mesh.update_faces(temp_mesh.nondegenerate_faces())
            temp_mesh.remove_unreferenced_vertices()
            temp_mesh.fix_normals()

            # Update vertices and faces
            mesh_points = np.array(temp_mesh.vertices)
            faces = np.array(temp_mesh.faces)

            # Remap colors
            # Note: This is simplified - in production you'd properly remap vertex colors
            if len(vertex_colors) != len(mesh_points):
                vertex_colors = vertex_colors[: len(mesh_points)]

        # Create PLY
        num_verts = len(mesh_points)
        num_faces = len(faces)

        vertex_dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
        vertex_data = np.zeros(num_verts, dtype=vertex_dtype)

        for i in range(num_verts):
            vertex_data[i] = (
                mesh_points[i, 0],
                mesh_points[i, 1],
                mesh_points[i, 2],
                int(vertex_colors[i, 0] * 255),
                int(vertex_colors[i, 1] * 255),
                int(vertex_colors[i, 2] * 255),
            )

        face_dtype = [("vertex_indices", "i4", (3,))]
        face_data = np.array([(face,) for face in faces], dtype=face_dtype)

        vertex_el = PlyElement.describe(vertex_data, "vertex")
        face_el = PlyElement.describe(face_data, "face")

        ply_data = PlyData([vertex_el, face_el])
        ply_data.write(output_path)
        print(f"Saved {self.metric_name} mesh to {output_path}")

        self.mesh_path = output_path
        return output_path

    def render_frame(
        self,
        yaw: float,
        pitch: float,
        cfg: str = "Head",
        resolution: Tuple[int, int] = (256, 256),
    ) -> Optional[np.ndarray]:
        """Render the mesh from a specific viewing angle"""
        if not self.mesh_path or not os.path.exists(self.mesh_path):
            return self._create_metric_visualization(yaw, pitch, resolution)

        # Try rendering methods - prefer PyRender for headless environments
        if PYRENDER_AVAILABLE and TRIMESH_AVAILABLE:
            try:
                return self._render_frame_pyrender(yaw, pitch, cfg, resolution)
            except Exception as e:
                print(f"PyRender failed: {e}, trying Open3D...")

        if OPEN3D_AVAILABLE and False:  # Disabled Open3D due to headless issues
            try:
                return self._render_frame_open3d(yaw, pitch, cfg, resolution)
            except Exception as e:
                print(f"Open3D rendering failed: {e}, using fallback...")

        # Fallback: Create a simple visualization based on metric values
        return self._create_metric_visualization(yaw, pitch, resolution)

    def _create_metric_visualization(
        self, yaw: float, pitch: float, resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Create a simple visualization when 3D rendering is not available"""
        # Create a circular heatmap showing metric value at this angle
        frame = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 240

        # Get metric value for this angle
        yaw_deg = int(np.degrees(yaw) % 360)
        pitch_deg = int(np.degrees(pitch))

        # Find closest metric value
        min_dist = float("inf")
        metric_value = 0.5

        for (y_deg, p_deg), value in self.metric_values_by_angle.items():
            dist = np.sqrt((y_deg - yaw_deg) ** 2 + (p_deg - pitch_deg) ** 2)
            if dist < min_dist:
                min_dist = dist
                metric_value = value

        # Create circular gradient based on metric value
        center_x, center_y = resolution[0] // 2, resolution[1] // 2
        radius = min(resolution) // 3

        # Use magma colormap for consistency
        import matplotlib.cm as cm

        cmap = cm.get_cmap("magma")
        rgba = cmap(metric_value)
        color = np.array(rgba[:3]) * 255

        # Draw filled circle with the color
        cv2.circle(frame, (center_x, center_y), radius, color.tolist(), -1)

        # Add text
        cv2.putText(
            frame,
            f"{self.metric_name}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            frame,
            f"Value: {metric_value:.3f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            frame,
            f"Yaw: {yaw_deg}° Pitch: {pitch_deg}°",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            1,
        )

        return frame

    def _render_frame_pyrender(
        self, yaw: float, pitch: float, cfg: str, resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Render using pyrender"""
        import pyrender

        # Set up headless rendering
        os.environ["PYOPENGL_PLATFORM"] = "egl"  # or 'osmesa'

        # Load mesh
        mesh = trimesh.load(self.mesh_path)

        # Ensure mesh has vertex colors
        if (
            not hasattr(mesh.visual, "vertex_colors")
            or mesh.visual.vertex_colors is None
        ):
            print("Warning: Mesh missing vertex colors")

        # Create scene
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0])
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(mesh_pyrender)

        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = np.eye(4)
        light_pose[:3, 2] = [0.3, -0.5, -1.0]
        scene.add(light, pose=light_pose)

        # Camera setup - adjusted for better FOV match
        camera_lookat_point = (
            np.array([0, 0, 0.2]) if cfg == "FFHQ" else np.array([0, 0, 0])
        )
        radius = 1.8  # Further reduced from 2.2 for even closer view

        # Convert spherical to Cartesian
        cam_x = radius * np.sin(pitch) * np.cos(yaw)
        cam_y = radius * np.cos(pitch)
        cam_z = radius * np.sin(pitch) * np.sin(yaw)
        cam_pos = np.array([cam_x, cam_y, cam_z]) + camera_lookat_point

        # Create camera matrix
        forward = camera_lookat_point - cam_pos
        forward = forward / np.linalg.norm(forward)
        up_temp = np.array([0, 1, 0])
        right = np.cross(forward, up_temp)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward
        camera_pose[:3, 3] = cam_pos

        # Create camera and render - adjusted FOV for better match
        camera = pyrender.PerspectiveCamera(
            yfov=np.radians(21.0)
        )  # Further increased from 25.0 for wider FOV
        scene.add(camera, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, _ = renderer.render(scene)
        renderer.delete()

        # Flip horizontally to match original video orientation
        color = np.fliplr(color)

        return color

    def _render_frame_open3d(
        self, yaw: float, pitch: float, cfg: str, resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Render using Open3D"""
        try:
            # Load mesh
            mesh = o3d.io.read_triangle_mesh(self.mesh_path)

            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=resolution[0], height=resolution[1])
            vis.add_geometry(mesh)

            # Set rendering options - with error handling
            try:
                opt = vis.get_render_option()
                if opt is not None:
                    opt.background_color = np.array([1, 1, 1])
                    opt.mesh_show_wireframe = False
                    opt.mesh_show_back_face = True
            except Exception as e:
                print(f"Warning: Could not set render options: {e}")

            # Camera setup
            ctr = vis.get_view_control()
            if ctr is None:
                raise RuntimeError(
                    "Open3D view control not available - may need display or different Open3D version"
                )

            camera_lookat_point = (
                np.array([0, 0, 0.2]) if cfg == "FFHQ" else np.array([0, 0, 0])
            )
            radius = 1.8  # Further reduced from 2.2 for even closer view

            # Convert spherical to Cartesian
            cam_x = radius * np.sin(pitch) * np.cos(yaw)
            cam_y = radius * np.cos(pitch)
            cam_z = radius * np.sin(pitch) * np.sin(yaw)
            cam_pos = np.array([cam_x, cam_y, cam_z]) + camera_lookat_point

            # Set camera - with error checking
            if hasattr(ctr, "set_lookat"):
                ctr.set_lookat(camera_lookat_point)
                ctr.set_up([0, 1, 0])
                ctr.set_front(camera_lookat_point - cam_pos)
                ctr.set_zoom(0.9)  # Increased from 0.7 for closer view
            else:
                print("Warning: Open3D camera control methods not available")

            # Render
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            img = vis.capture_screen_float_buffer(do_render=True)
            img = np.asarray(img)
            img = (img * 255).astype(np.uint8)

            # Flip horizontally to match original video orientation
            img = np.fliplr(img)

            vis.destroy_window()

            return img

        except Exception as e:
            print(f"Error in Open3D rendering: {e}")
            # Return a placeholder image
            placeholder = (
                np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 200
            )
            cv2.putText(
                placeholder,
                f"Render Error: {str(e)[:30]}",
                (10, resolution[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )
            return placeholder


class SphereHeadGUARD3DMeshEvaluator:
    """Evaluator with 3D mesh visualization for each metric"""

    def __init__(
        self,
        spherehead_model_path: str,
        guard_script_path: str,
        output_dir: str = "experiments_3dmesh",
        device: Optional[str] = None,
        voxel_res: int = 128,
        sdf_batch_size: int = 128,
        vertex_batch_size: int = 128,
        guard_batch_size: int = 128,
        viz_batch_size: int = 1,
    ):
        """Initialize the evaluator."""
        self.spherehead_model_path = spherehead_model_path
        self.guard_script_path = guard_script_path
        self.output_dir = output_dir
        self.voxel_res = voxel_res
        self.sdf_batch_size = sdf_batch_size
        self.vertex_batch_size = vertex_batch_size
        self.guard_batch_size = guard_batch_size
        self.viz_batch_size = viz_batch_size

        # Cache for plot data to avoid recomputation
        self.plot_cache = {}

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = get_best_gpu_device()

        # Load SphereHead model
        self._load_spherehead_model()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Storage for camera poses and generated data
        self.camera_poses = []  # List of (yaw, pitch) tuples
        self.generated_frames = []
        self.w_latent = None  # Store w for 3D reconstruction

    def _load_spherehead_model(self):
        """Load the SphereHead generator model"""
        print(f"Loading SphereHead model from {self.spherehead_model_path}...")

        with dnnlib.util.open_url(self.spherehead_model_path) as f:
            data = load_network_pkl_cpu_safe(f, self.device)
            self.G = data["G_ema"].to(self.device)

        # Configure model
        self.G.rendering_kwargs["white_back"] = True
        self.G.neural_rendering_resolution = 300

        # Reload modules
        G_new = (
            TriPlaneGenerator(*self.G.init_args, **self.G.init_kwargs)
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        misc.copy_params_and_buffers(self.G, G_new, require_all=True)
        G_new.neural_rendering_resolution = self.G.neural_rendering_resolution
        G_new.rendering_kwargs = self.G.rendering_kwargs
        self.G = G_new

        print("SphereHead model loaded successfully")

    def generate_synthetic_video_with_tracking(
        self,
        seed: int,
        num_frames: int = 90,
        output_path: str = None,
        cfg: str = "Head",
        truncation_psi: float = 0.5,
        exp_dir: str = None,
    ) -> Dict[str, Any]:
        """
        Generate synthetic video while tracking camera poses for later mesh rendering.
        """
        print(f"\nGenerating synthetic video with seed={seed}, frames={num_frames}")

        # Set up paths
        if output_path is None:
            if exp_dir is None:
                exp_dir = os.path.join(self.output_dir, cfg, str(seed))
                os.makedirs(exp_dir, exist_ok=True)
            output_path = os.path.join(exp_dir, "original_video.mp4")

        # Generate latent code for consistent identity
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(
            self.device
        )

        # Camera parameters
        camera_lookat_point = (
            torch.tensor([0, 0, 0.2], device=self.device)
            if cfg == "FFHQ"
            else torch.tensor([0, 0, 0], device=self.device)
        )
        intrinsics = torch.tensor(
            [[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=self.device
        )

        # Generate initial w vector for consistent identity
        pose_cond_rad = 90 / 180 * np.pi
        initial_cam2world = LookAtPoseSampler.sample(
            pose_cond_rad, 3.14 / 2, camera_lookat_point, radius=2.7, device=self.device
        )
        initial_c = torch.cat(
            [initial_cam2world.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
        )
        self.w_latent = self.G.mapping(
            z=z, c=initial_c, truncation_psi=truncation_psi, truncation_cutoff=14
        )

        # Clear previous tracking data
        self.camera_poses = []
        self.generated_frames = []

        # Video writers
        writer = imageio.get_writer(output_path, fps=30)
        depth_path = output_path.replace(".mp4", "_depth.mp4")
        depth_writer = imageio.get_writer(depth_path, fps=30)

        for frame_idx in tqdm(range(num_frames), desc="Generating frames"):
            # Camera motion
            if cfg == "Head":
                pitch_range = 0.5
                yaw = 3.14 / 2 + 2 * 3.14 * frame_idx / num_frames
                pitch = (
                    3.14 / 2
                    - 0.05
                    + pitch_range * np.sin(2 * 3.14 * frame_idx / num_frames)
                )
            else:
                pitch_range = 0.25
                yaw_range = 1.5
                yaw = 3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / num_frames)
                pitch = (
                    3.14 / 2
                    - 0.05
                    + pitch_range * np.cos(2 * 3.14 * frame_idx / num_frames)
                )

            # Store camera pose
            self.camera_poses.append((yaw, pitch))

            # Generate camera pose
            cam2world_pose = LookAtPoseSampler.sample(
                yaw, pitch, camera_lookat_point, radius=2.7, device=self.device
            )
            c_synth = torch.cat(
                [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
            )

            # Generate image
            with torch.no_grad():
                synthesis_result = self.G.synthesis(
                    ws=self.w_latent, c=c_synth, noise_mode="const"
                )
                img = synthesis_result["image"][0]

                # Extract depth
                if "image_depth" in synthesis_result:
                    depth = synthesis_result["image_depth"][0]
                    depth = -depth
                    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    depth_img = depth.clamp(0, 1).cpu().numpy()
                    if len(depth_img.shape) == 2:
                        depth_img = np.expand_dims(depth_img, 0)
                    depth_img = np.repeat(depth_img, 3, axis=0)
                    depth_img = np.transpose(depth_img, (1, 2, 0))
                    depth_img = (depth_img * 255).astype(np.uint8)
                    depth_writer.append_data(depth_img)

                # Handle white background
                if "image_mask" in synthesis_result:
                    mask = synthesis_result["image_mask"][0]
                    if mask.shape[-2:] != img.shape[-2:]:
                        mask = torch.nn.functional.interpolate(
                            mask.unsqueeze(0),
                            size=img.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    if mask.shape[0] == 1:
                        mask = mask.repeat(3, 1, 1)
                    white_bg = torch.ones_like(img)
                    img = img * mask + white_bg * (1 - mask)

                # Convert to RGB
                img_np = img.clamp(-1, 1).cpu().numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np + 1) / 2 * 255
                img_np = img_np.astype(np.uint8)

                self.generated_frames.append(img_np)
                writer.append_data(img_np)

        writer.close()
        depth_writer.close()

        print(f"Generated video saved to {output_path}")
        print(f"Depth video saved to {depth_path}")

        return {
            "video_path": output_path,
            "depth_path": depth_path,
            "num_frames": num_frames,
            "seed": seed,
            "fps": 30,
            "cfg": cfg,
        }

    def extract_3d_shape(self, exp_dir: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """Extract 3D shape using SDF"""
        print("\nExtracting 3D shape from latent representation...")

        # Create samples
        samples, voxel_origin, voxel_size = create_samples(
            N=self.voxel_res,
            voxel_origin=[0, 0, 0],
            cube_length=self.G.rendering_kwargs["box_warp"],
        )
        samples = samples.to(self.device)

        sigmas = torch.zeros(
            (samples.shape[0], samples.shape[1], 1), device=self.device
        )
        transformed_ray_directions_expanded = torch.zeros(
            (samples.shape[0], samples.shape[1], 3), device=self.device
        )
        transformed_ray_directions_expanded[..., -1] = -1

        # Extract SDF in batches - use larger batches for SDF extraction
        if self.sdf_batch_size == 128:
            # If using default 128, use larger batch for SDF specifically
            max_batch = 500000 if self.device.type == "cuda" else 50000
        else:
            max_batch = (
                self.sdf_batch_size
                if self.device.type == "cuda"
                else min(self.sdf_batch_size // 10, 50000)
            )
        head = 0

        with torch.no_grad():
            pbar = tqdm(total=samples.shape[1], desc="Extracting SDF")
            while head < samples.shape[1]:
                batch_size = min(max_batch, samples.shape[1] - head)
                sigma = self.G.sample_mixed(
                    samples[:, head : head + batch_size],
                    transformed_ray_directions_expanded[:, :batch_size],
                    self.w_latent,
                    truncation_psi=0.5,
                    noise_mode="const",
                )["sigma"]
                sigmas[:, head : head + batch_size] = sigma
                head += batch_size
                pbar.update(batch_size)
            pbar.close()

        sigmas = (
            sigmas.reshape((self.voxel_res, self.voxel_res, self.voxel_res))
            .cpu()
            .numpy()
        )
        sigmas = np.flip(sigmas, 0)

        # Apply padding to remove artifacts
        pad = int(30 * self.voxel_res / 256)
        pad_top = int(38 * self.voxel_res / 256)
        sigmas[:pad] = 0
        sigmas[-pad:] = 0
        sigmas[:, :pad] = 0
        sigmas[:, -pad_top:] = 0
        sigmas[:, :, :pad] = 0
        sigmas[:, :, -pad:] = 0

        # Save raw shape data
        shape_data = {
            "sigmas": sigmas,
            "voxel_origin": voxel_origin,
            "voxel_size": voxel_size,
        }

        np.save(os.path.join(exp_dir, "3d_shape_data.npy"), shape_data)

        return sigmas, voxel_origin, voxel_size

    def run_guard_anonymization(
        self, video_path: str, output_path: str = None
    ) -> Dict[str, str]:
        """Run GUARD anonymization on a video."""
        print(f"\nRunning GUARD anonymization on {video_path}")

        # Set up paths
        if output_path is None:
            base_dir = os.path.dirname(video_path)
            output_path = os.path.join(base_dir, "anonymized_video.mp4")

        # Get GUARD directory
        guard_dir = os.path.dirname(os.path.dirname(self.guard_script_path))
        script_relative_path = os.path.relpath(self.guard_script_path, guard_dir)

        # Construct command
        cmd = [
            sys.executable,
            script_relative_path,
            "--input_video",
            os.path.abspath(video_path),
            "--output_video",
            os.path.abspath(output_path),
            "--batch_size",
            str(self.guard_batch_size),
        ]

        # Run anonymization
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=guard_dir
            )
            print("GUARD anonymization completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running GUARD: {e}")
            raise

        metrics_path = output_path.replace(".mp4", "_metrics.json")

        return {"anonymized_path": output_path, "metrics_path": metrics_path}

    def extract_anonymized_frames(
        self, guard_video_path: str, original_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Extract anonymized frames from GUARD output"""
        print("Extracting anonymized frames from GUARD output...")

        h, w = original_shape
        reader = imageio.get_reader(guard_video_path)
        frames = []

        for frame in reader:
            # Extract anonymized portion
            if frame.shape[0] > h:  # Has plot below
                video_portion = frame[:h, :]
                if video_portion.shape[1] > w:  # Side-by-side
                    anon_frame = video_portion[:, w : 2 * w]
                else:
                    anon_frame = video_portion
            else:
                if frame.shape[1] > w:  # Side-by-side
                    anon_frame = frame[:, w : 2 * w]
                else:
                    anon_frame = frame

            if anon_frame.shape[:2] != (h, w):
                anon_frame = cv2.resize(anon_frame, (w, h))

            frames.append(anon_frame)

        reader.close()
        return frames

    def create_metric_meshes(
        self,
        metrics: Dict[str, Any],
        sigmas: np.ndarray,
        voxel_origin: np.ndarray,
        voxel_size: float,
        exp_dir: str,
    ) -> Dict[str, MetricMeshRenderer]:
        """Create 3D meshes colored by each metric"""
        print(f"\nCreating 3D meshes for {len(metrics['metric_keys'])} metrics...")

        mesh_renderers = {}

        for metric_name in metrics["metric_keys"]:
            print(f"\nProcessing metric: {metric_name}")

            # Create renderer for this metric with batch size
            renderer = MetricMeshRenderer(
                metric_name, self.device, self.vertex_batch_size
            )

            # Populate metric values from all frames
            for frame_idx, frame_metrics in enumerate(metrics["frame_metrics"]):
                if frame_idx < len(self.camera_poses):
                    yaw, pitch = self.camera_poses[frame_idx]
                    value = frame_metrics.get(metric_name, 0)
                    # Handle NaN values
                    if np.isnan(value):
                        value = 0
                    renderer.add_metric_value(yaw, pitch, value)

            # Create colored mesh
            mesh_path = os.path.join(exp_dir, f"mesh_{metric_name}.ply")

            # Use consistent colormap for all metrics
            colormap = "magma"

            renderer.create_metric_colored_mesh(
                sigmas, voxel_origin, voxel_size, mesh_path, colormap
            )

            mesh_renderers[metric_name] = renderer

        return mesh_renderers

    def add_text_label(self, image, text, x, y, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
        """Add text label with background to image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Add padding
        padding = 5
        
        # Draw background rectangle
        cv2.rectangle(image, 
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     bg_color, -1)
        
        # Draw text
        cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
        
        return image

    def create_multi_row_3d_visualization(
        self,
        original_frames: List[np.ndarray],
        depth_frames: List[np.ndarray],
        anonymized_frames: List[np.ndarray],
        metrics: Dict[str, Any],
        mesh_renderers: Dict[str, MetricMeshRenderer],
        output_path: str,
        cfg: str = "Head",
    ):
        """Create visualization with 3D mesh heatmaps for each metric"""
        print("\nCreating multi-row 3D visualization...")

        # Clear GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            clear_gpu_memory(self.device)

        # Disable matplotlib interactive mode
        plt.ioff()

        # Layout configuration
        panel_width = 256
        panel_height = 256
        plot_width = 400

        # Calculate dimensions - now with 5 columns (original, anonymized, depth, diff heatmap, 3D mesh, plot)
        num_metrics = len(metrics["metric_keys"])
        total_width = panel_width * 5 + plot_width
        total_height = panel_height * num_metrics

        # Create video writer
        writer = imageio.get_writer(
            output_path, fps=30, codec="libx264", pixelformat="yuv420p"
        )

        # Process frames in batches
        total_frames = len(original_frames)

        with tqdm(total=total_frames, desc="Creating 3D visualization") as pbar:
            for batch_start in range(0, total_frames, self.viz_batch_size):
                batch_end = min(batch_start + self.viz_batch_size, total_frames)
                batch_frames = []

                # Pre-render all mesh frames for this batch
                mesh_frames_batch = {metric: [] for metric in metrics["metric_keys"]}

                # Batch render mesh frames
                for frame_idx in range(batch_start, batch_end):
                    yaw, pitch = self.camera_poses[frame_idx]

                    for metric_name in metrics["metric_keys"]:
                        renderer = mesh_renderers[metric_name]
                        mesh_frame = renderer.render_frame(
                            yaw, pitch, cfg, (panel_width, panel_height)
                        )
                        mesh_frames_batch[metric_name].append(mesh_frame)

                # Process each frame in the batch
                for batch_idx, frame_idx in enumerate(range(batch_start, batch_end)):
                    # Get metrics for this frame
                    frame_metrics = metrics["frame_metrics"][frame_idx]

                    # Create composite frame
                    composite = np.zeros((total_height, total_width, 3), dtype=np.uint8)

                    # Pre-resize original, anonymized, and depth frames
                    orig_resized = cv2.resize(
                        original_frames[frame_idx], (panel_width, panel_height)
                    )
                    anon_resized = cv2.resize(
                        anonymized_frames[frame_idx], (panel_width, panel_height)
                    )
                    depth_resized = cv2.resize(
                        depth_frames[frame_idx], (panel_width, panel_height)
                    )
                    
                    # Create difference heatmap
                    diff = np.abs(orig_resized.astype(np.float32) - anon_resized.astype(np.float32))
                    diff_gray = np.mean(diff, axis=2)  # Convert to grayscale
                    diff_normalized = (diff_gray / 255.0 * 255).astype(np.uint8)
                    
                    # Apply colormap to difference
                    diff_heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
                    diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)

                    # Process each metric
                    for row_idx, metric_name in enumerate(metrics["metric_keys"]):
                        y_start = row_idx * panel_height
                        y_end = y_start + panel_height

                        # 1. Original frame (pre-resized)
                        composite[y_start:y_end, 0:panel_width] = orig_resized
                        # Add label only on first metric row
                        if row_idx == 0:
                            self.add_text_label(composite, "Original", 10, y_start + 25)

                        # 2. Anonymized frame
                        composite[y_start:y_end, panel_width:2*panel_width] = anon_resized
                        if row_idx == 0:
                            self.add_text_label(composite, "Anonymized", panel_width + 10, y_start + 25)

                        # 3. Depth map (pre-resized)
                        composite[
                            y_start:y_end, 2*panel_width : 3 * panel_width
                        ] = depth_resized
                        if row_idx == 0:
                            self.add_text_label(composite, "Depth Map", 2*panel_width + 10, y_start + 25)

                        # 4. Difference heatmap
                        composite[
                            y_start:y_end, 3*panel_width : 4 * panel_width
                        ] = diff_heatmap
                        if row_idx == 0:
                            self.add_text_label(composite, "Difference", 3*panel_width + 10, y_start + 25)

                        # 5. 3D mesh rendered from current viewpoint (from batch)
                        mesh_frame = mesh_frames_batch[metric_name][batch_idx]
                        if mesh_frame is not None:
                            composite[
                                y_start:y_end, 4 * panel_width : 5 * panel_width
                            ] = mesh_frame
                        if row_idx == 0:
                            self.add_text_label(composite, "3D Heatmap", 4*panel_width + 10, y_start + 25)

                        # 6. Metric plot
                        metric_value = frame_metrics.get(metric_name, 0)
                        # Handle NaN values
                        if np.isnan(metric_value):
                            metric_value = 0
                        plot_frame = self.create_metric_plot(
                            metric_name,
                            frame_idx,
                            metric_value,
                            metrics["frame_metrics"][: frame_idx + 1],
                            plot_width,
                            panel_height,
                        )
                        composite[y_start:y_end, 5 * panel_width :] = plot_frame
                        if row_idx == 0:
                            self.add_text_label(composite, "Metric Plot", 5*panel_width + 10, y_start + 25)

                    writer.append_data(composite)

                pbar.update(batch_end - batch_start)

        writer.close()
        print(f"3D visualization saved to {output_path}")

    def create_metric_plot(
        self,
        metric_name: str,
        frame_idx: int,
        current_value: float,
        frame_metrics: List[Dict],
        plot_width: int,
        plot_height: int,
    ) -> np.ndarray:
        """Create a plot for a single metric with caching"""
        # Check cache
        cache_key = f"{metric_name}_{frame_idx}"
        if cache_key in self.plot_cache:
            return self.plot_cache[cache_key]

        # Use non-interactive backend for better performance
        import matplotlib

        matplotlib.use("Agg")

        # Don't close all figures, just create new one
        fig = plt.figure(
            figsize=(plot_width / 100, plot_height / 100), dpi=100, facecolor="black"
        )
        ax = fig.add_subplot(111)

        # Style (simplified)
        ax.set_facecolor("#1a1a1a")
        ax.grid(True, alpha=0.3, color="gray", linewidth=0.5)

        # Extract values efficiently using list comprehension - handle missing/NaN as 0
        metric_data = [
            (i, fm.get(metric_name, 0) if not np.isnan(fm.get(metric_name, 0)) else 0)
            for i, fm in enumerate(frame_metrics)
        ]

        if metric_data:
            frames, values = zip(*metric_data)

            # Plot with minimal overhead
            ax.plot(frames, values, "cyan", linewidth=1.5)
            ax.scatter([frame_idx], [current_value], color="red", s=30, zorder=5)

        # Labels (simplified)
        ax.set_xlabel("Frame", color="white", fontsize=8)
        ax.set_ylabel(
            metric_name[:15], color="white", fontsize=6
        )  # Truncate long names
        ax.set_xlim(0, max(frame_idx + 5, 30))

        # Skip some styling for speed
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors="white", labelsize=6)

        # Current value annotation (simplified)
        ax.text(
            0.98,
            0.95,
            f"{current_value:.3f}",
            transform=ax.transAxes,
            color="cyan",
            fontsize=8,
            ha="right",
            va="top",
        )

        # Faster rendering
        plt.tight_layout(pad=0.5)

        # Render directly without canvas wrapper
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_image = np.ascontiguousarray(buf)

        # Close figure immediately
        plt.close(fig)

        # Cache for reuse (limit cache size)
        if len(self.plot_cache) < 100:
            self.plot_cache[cache_key] = plot_image

        return plot_image

    def create_compact_3d_visualization(
        self,
        anonymized_frames: List[np.ndarray],
        original_frames: List[np.ndarray],
        depth_frames: List[np.ndarray],
        metrics: Dict[str, Any],
        mesh_renderers: Dict[str, MetricMeshRenderer],
        output_path: str,
        cfg: str = "Head",
    ):
        """Create compact visualization with anonymized/diff/heatmap on top and 6 metrics in 2x3 grid"""
        print("\nCreating compact 3D visualization...")

        # Clear GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            clear_gpu_memory(self.device)

        # Disable matplotlib interactive mode
        plt.ioff()

        # Layout configuration
        panel_width = 320  # Slightly larger for better visibility
        panel_height = 320
        
        # Top row: 3 panels (anonymized, difference, 3D heatmap)
        # Bottom rows: 2x3 grid of 6 metrics
        top_row_height = panel_height
        metric_row_height = int(panel_height * 0.8)  # Increased height for better plot visibility
        
        total_width = panel_width * 3
        total_height = top_row_height + metric_row_height * 2
        
        # Create video writer
        writer = imageio.get_writer(
            output_path, fps=30, codec="libx264", pixelformat="yuv420p"
        )
        
        # Process frames
        total_frames = len(original_frames)
        
        # We'll use the first metric as the reference for the 3D heatmap
        reference_metric = metrics["metric_keys"][0] if metrics["metric_keys"] else None
        
        with tqdm(total=total_frames, desc="Creating compact visualization") as pbar:
            for frame_idx in range(total_frames):
                # Create composite frame
                composite = np.zeros((total_height, total_width, 3), dtype=np.uint8)
                
                # Get metrics for this frame
                frame_metrics = metrics["frame_metrics"][frame_idx]
                
                # Pre-resize frames
                anon_resized = cv2.resize(anonymized_frames[frame_idx], (panel_width, panel_height))
                orig_resized = cv2.resize(original_frames[frame_idx], (panel_width, panel_height))
                
                # Top row
                # 1. Anonymized video
                composite[0:top_row_height, 0:panel_width] = anon_resized
                self.add_text_label(composite, "Anonymized", 10, 25)
                
                # 2. Difference heatmap
                diff = np.abs(orig_resized.astype(np.float32) - anon_resized.astype(np.float32))
                diff_gray = np.mean(diff, axis=2)
                diff_normalized = (diff_gray / 255.0 * 255).astype(np.uint8)
                diff_heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
                diff_heatmap = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)
                composite[0:top_row_height, panel_width:2*panel_width] = diff_heatmap
                self.add_text_label(composite, "Difference", panel_width + 10, 25)
                
                # 3. 3D Heatmap (using first metric)
                if reference_metric and reference_metric in mesh_renderers:
                    yaw, pitch = self.camera_poses[frame_idx]
                    renderer = mesh_renderers[reference_metric]
                    mesh_frame = renderer.render_frame(yaw, pitch, cfg, (panel_width, panel_height))
                    if mesh_frame is not None:
                        composite[0:top_row_height, 2*panel_width:3*panel_width] = mesh_frame
                    self.add_text_label(composite, f"3D Heatmap ({reference_metric[:20]})", 2*panel_width + 10, 25)
                
                # Bottom rows: 2x3 grid of metrics
                metric_panel_width = panel_width
                metric_panel_height = metric_row_height
                
                for idx, metric_name in enumerate(metrics["metric_keys"][:6]):
                    row = idx // 3
                    col = idx % 3
                    
                    y_start = top_row_height + row * metric_panel_height
                    y_end = y_start + metric_panel_height
                    x_start = col * metric_panel_width
                    x_end = x_start + metric_panel_width
                    
                    # Get metric value
                    metric_value = frame_metrics.get(metric_name, 0)
                    if np.isnan(metric_value):
                        metric_value = 0
                    
                    # Create metric plot
                    plot_frame = self.create_metric_plot(
                        metric_name,
                        frame_idx,
                        metric_value,
                        metrics["frame_metrics"][:frame_idx + 1],
                        metric_panel_width,
                        metric_panel_height,
                    )
                    
                    # Resize plot frame if needed
                    if plot_frame.shape[:2] != (metric_panel_height, metric_panel_width):
                        plot_frame = cv2.resize(plot_frame, (metric_panel_width, metric_panel_height))
                    
                    composite[y_start:y_end, x_start:x_end] = plot_frame
                    
                    # Add metric name label
                    label_text = metric_name[:25]  # Truncate long names
                    self.add_text_label(composite, label_text, x_start + 10, y_start + 25)
                
                writer.append_data(composite)
                pbar.update(1)
        
        writer.close()
        print(f"Compact visualization saved to {output_path}")

    def run_experiment(self, seed: int, num_frames: int = 90, cfg: str = "Head"):
        """Run complete experiment with 3D mesh visualizations"""
        print(f"\n{'='*60}")
        print(f"Running 3D mesh experiment for seed {seed}, cfg {cfg}")
        print(f"{'='*60}")

        # Create experiment directory with new structure
        exp_dir = os.path.join(self.output_dir, cfg, str(seed))
        os.makedirs(exp_dir, exist_ok=True)

        # Step 1: Generate synthetic video with tracking
        video_info = self.generate_synthetic_video_with_tracking(seed, num_frames, cfg=cfg, exp_dir=exp_dir)

        # Step 2: Extract 3D shape
        sigmas, voxel_origin, voxel_size = self.extract_3d_shape(exp_dir)

        # Step 3: Run GUARD anonymization
        guard_results = self.run_guard_anonymization(video_info["video_path"])

        # Step 4: Extract frames
        depth_reader = imageio.get_reader(video_info["depth_path"])
        depth_frames = [frame for frame in depth_reader]
        depth_reader.close()

        anonymized_frames = self.extract_anonymized_frames(
            guard_results["anonymized_path"], (512, 512)
        )

        # Step 5: Parse metrics
        with open(guard_results["metrics_path"], "r") as f:
            metrics_data = json.load(f)

        # Get numeric metrics
        frame_metrics = metrics_data["frame_metrics"]
        metric_keys = set()
        for frame in frame_metrics:
            metric_keys.update(frame.keys())

        exclude_keys = {"frame_index", "timestamp", "face_detected"}
        metric_keys = metric_keys - exclude_keys

        # Filter to numeric metrics, excluding FAR metrics
        numeric_metrics = []
        for key in metric_keys:
            # Skip FAR metrics
            if "FAR@" in key and "_verified" in key:
                continue
            sample_values = [
                frame.get(key) for frame in frame_metrics[:10] if key in frame
            ]
            if sample_values and all(
                isinstance(v, (int, float, bool)) for v in sample_values
            ):
                numeric_metrics.append(key)

        metrics = {
            "frame_metrics": frame_metrics,
            "metric_keys": sorted(numeric_metrics),  # Include all non-FAR metrics
        }

        print(
            f"\nVisualizing {len(metrics['metric_keys'])} metrics (excluding FAR metrics):"
        )
        for i, metric in enumerate(metrics["metric_keys"]):
            print(f"  {i+1}. {metric}")

        # Step 6: Create metric-colored meshes
        mesh_renderers = self.create_metric_meshes(
            metrics, sigmas, voxel_origin, voxel_size, exp_dir
        )

        # Step 7: Create final visualization
        final_output = os.path.join(exp_dir, "3dmesh_visualization.mp4")
        self.create_multi_row_3d_visualization(
            self.generated_frames,
            depth_frames,
            anonymized_frames,
            metrics,
            mesh_renderers,
            final_output,
            video_info["cfg"],
        )

        # Step 8: Create compact visualization
        compact_output = os.path.join(exp_dir, "3dmesh_compact_visualization.mp4")
        # Select top 6 metrics for compact view
        top_metrics = metrics["metric_keys"][:6] if len(metrics["metric_keys"]) >= 6 else metrics["metric_keys"]
        self.create_compact_3d_visualization(
            anonymized_frames,
            self.generated_frames,
            depth_frames,
            {"frame_metrics": metrics["frame_metrics"], "metric_keys": top_metrics},
            mesh_renderers,
            compact_output,
            video_info["cfg"],
        )

        print(f"\nExperiment complete! Results saved in {exp_dir}")
        print(f"3D mesh visualization: {final_output}")
        print(f"Compact visualization: {compact_output}")

        # Save summary
        summary = {
            "seed": seed,
            "cfg": cfg,
            "num_frames": num_frames,
            "metrics_visualized": metrics["metric_keys"],
            "mesh_files": [f"mesh_{m}.ply" for m in metrics["metric_keys"]],
            "voxel_resolution": self.voxel_res,
        }

        with open(os.path.join(exp_dir, "experiment_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SphereHead-GUARD 3D Mesh Evaluation")

    parser.add_argument(
        "--spherehead_model", type=str, default="../models/spherehead-ckpt-025000.pkl"
    )
    parser.add_argument(
        "--guard_script",
        type=str,
        default="GUARD/video_anonymization/video_anonymization_FIVA_temporal_metrics.py",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=90)
    parser.add_argument("--cfg", type=str, default="Head", choices=["Head", "FFHQ", "Cats"], help="Configuration type")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--voxel_res",
        type=int,
        default=128,
        help="Voxel resolution for 3D shape extraction",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for computation (default: 128)",
    )
    parser.add_argument(
        "--sdf_batch_size",
        type=int,
        default=None,
        help="Batch size for SDF extraction (default: 128)",
    )
    parser.add_argument(
        "--vertex_batch_size",
        type=int,
        default=None,
        help="Batch size for vertex coloring (default: 128)",
    )
    parser.add_argument(
        "--viz_batch_size",
        type=int,
        default=1,
        help="Batch size for visualization frame processing (default: 1)",
    )

    args = parser.parse_args()

    # Set default batch sizes if not specified
    if args.sdf_batch_size is None:
        args.sdf_batch_size = 128
    if args.vertex_batch_size is None:
        args.vertex_batch_size = 128

    # Check model paths
    if not os.path.exists(args.spherehead_model):
        print(f"Error: SphereHead model not found at {args.spherehead_model}")
        sys.exit(1)

    if not os.path.exists(args.guard_script):
        print(f"Error: GUARD script not found at {args.guard_script}")
        sys.exit(1)

    # Create evaluator
    evaluator = SphereHeadGUARD3DMeshEvaluator(
        args.spherehead_model,
        args.guard_script,
        args.output_dir,
        args.device,
        args.voxel_res,
        args.sdf_batch_size,
        args.vertex_batch_size,
        args.batch_size,  # GUARD batch size
        args.viz_batch_size,  # Visualization batch size
    )

    # Run experiment
    evaluator.run_experiment(args.seed, args.num_frames, args.cfg)


if __name__ == "__main__":
    main()

