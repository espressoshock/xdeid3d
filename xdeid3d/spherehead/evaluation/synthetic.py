#!/usr/bin/env python3
"""
Synthetic Evaluation Tool for Neural Networks.
Generates synthetic samples with controllable characteristics (pose, gaze, etc.)
to evaluate target neural networks with known ground truth.
"""

import os
import sys
import torch
import numpy as np
import imageio
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import importlib.util
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available, mesh rendering will be skipped")

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print("Warning: pyrender not available, mesh rendering will use matplotlib fallback")
    
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

# Package imports (no sys.path manipulation needed)

from xdeid3d.spherehead import dnnlib
from xdeid3d.spherehead import legacy
from xdeid3d.spherehead.camera_utils import LookAtPoseSampler
from xdeid3d.spherehead.torch_utils import misc
from xdeid3d.spherehead.training.triplane import TriPlaneGenerator
from xdeid3d.spherehead.evaluation.gpu_utils import get_best_gpu_device, clear_gpu_memory
from xdeid3d.spherehead.shape_utils import convert_sdf_samples_to_ply

# Use the CPU-safe loading function from gen_videos
try:
    from xdeid3d.spherehead.scripts.gen_videos import load_network_pkl_cpu_safe
except ImportError:
    def load_network_pkl_cpu_safe(f, device):
        """Fallback CPU-safe loading."""
        import pickle
        import io
        buffer = io.BytesIO(f.read())
        try:
            buffer.seek(0)
            data = torch.load(buffer, map_location='cpu')
            if isinstance(data, dict) and 'G_ema' in data:
                return data
        except:
            pass
        buffer.seek(0)
        original_load = torch.load
        torch.load = lambda f, **kwargs: original_load(f, map_location='cpu', **kwargs)
        try:
            data = legacy.load_network_pkl(buffer)
        finally:
            torch.load = original_load
        return data


class BaseEvaluator(ABC):
    """
    Base class for neural network evaluators.
    
    Implement this class to create custom evaluators for synthetic samples.
    """
    
    def __init__(self, **kwargs):
        """Initialize evaluator with optional keyword arguments."""
        self.kwargs = kwargs
    
    @abstractmethod
    def evaluate(self, sample_data: Dict[str, Any]) -> float:
        """
        Evaluate a single sample/view.
        
        Args:
            sample_data: Dictionary containing:
                - 'image': The generated image (numpy array)
                - 'depth': Depth map if available (numpy array or None)
                - 'pose': Camera pose parameters (dict with 'yaw' and 'pitch')
                - 'seed': Random seed used
                - 'frame_idx': Frame index in sequence
        
        Returns:
            float: Score between 0 and 1 (higher is better)
        """
        pass
    
    def initialize(self, generator_config: Dict[str, Any]) -> None:
        """
        Optional initialization with generator configuration.
        
        Args:
            generator_config: Dictionary with generator settings
        """
        pass
    
    def finalize(self, all_evaluations: List[Dict[str, Any]]) -> None:
        """
        Optional finalization after all evaluations are complete.
        
        Args:
            all_evaluations: List of all evaluation results
        """
        pass


def fake_neural_network_evaluation(pose_params, seed=None):
    """
    Fake neural network evaluation that returns a metric based on pose.
    
    For demonstration, we create a metric that:
    - Is highest when viewing from the front (yaw ~= pi/2)
    - Decreases as we rotate away from front
    - Has some variation based on pitch
    """
    yaw = pose_params['yaw']
    pitch = pose_params['pitch']
    
    # Normalize yaw to [0, 2*pi]
    yaw_norm = yaw % (2 * np.pi)
    
    # Front view is around pi/2, back view is around 3*pi/2
    # Create a score that's highest at front view
    front_score = np.cos(yaw_norm - np.pi/2) ** 2
    
    # Add some pitch variation
    pitch_factor = 0.8 + 0.2 * np.cos(2 * (pitch - np.pi/2))
    
    # Add deterministic "noise" based on angle to make it more interesting
    # but consistent across calls
    angle_variation = 0.05 * np.sin(5 * yaw_norm) * np.cos(3 * pitch)
    
    # Final score in [0, 1]
    score = np.clip(front_score * pitch_factor + angle_variation, 0, 1)
    
    return score


def load_custom_evaluator(evaluator_path: str, evaluator_args: Dict[str, str]) -> BaseEvaluator:
    """
    Load a custom evaluator from a Python file.
    
    Args:
        evaluator_path: Path to Python file containing evaluator class
        evaluator_args: Dictionary of arguments to pass to evaluator
    
    Returns:
        Instance of the evaluator
    """
    # Load the module
    spec = importlib.util.spec_from_file_location("custom_evaluator", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    
    # Inject BaseEvaluator into the module's namespace so evaluators can use it
    module.BaseEvaluator = BaseEvaluator
    
    spec.loader.exec_module(module)
    
    # Find the evaluator class (should inherit from BaseEvaluator)
    evaluator_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseEvaluator) and obj != BaseEvaluator:
            evaluator_class = obj
            break
    
    if evaluator_class is None:
        raise ValueError(f"No evaluator class found in {evaluator_path}")
    
    # Create instance with arguments
    return evaluator_class(**evaluator_args)


def parse_key_value_pairs(args_list: List[str]) -> Dict[str, str]:
    """Parse key=value pairs from command line arguments."""
    result = {}
    for arg in args_list:
        if '=' in arg:
            key, value = arg.split('=', 1)
            result[key] = value
    return result


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    """Create 3D sampling grid for SDF extraction."""
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)
    
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N
    
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    
    return samples.unsqueeze(0), voxel_origin, voxel_size


def create_colored_mesh_ply(sigmas, voxel_origin, voxel_size, evaluations, output_path, colormap='viridis'):
    """
    Create a colored mesh based on evaluation scores.
    This creates a PLY file with vertex colors based on the fake neural network scores.
    """
    import skimage.measure
    from plyfile import PlyData, PlyElement
    from scipy import ndimage
    
    print("Generating colored heatmap mesh...")
    
    # Pre-process sigmas to reduce artifacts
    print("Pre-processing volume data...")
    
    # Apply slight Gaussian smoothing to reduce noise
    sigmas_smooth = ndimage.gaussian_filter(sigmas, sigma=0.5)
    
    # Ensure proper boundary conditions (pad with zeros)
    pad_width = 2
    sigmas_padded = np.pad(sigmas_smooth, pad_width, mode='constant', constant_values=0)
    
    # Extract mesh using marching cubes
    level = 10.0  # Isosurface level
    
    # Use step_size to control mesh resolution (smaller = higher quality)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            sigmas_padded, level=level, spacing=[voxel_size] * 3,
            step_size=1,  # Maximum quality
            allow_degenerate=False  # Prevent degenerate triangles
        )
        # Adjust vertices for padding
        verts -= pad_width * voxel_size
    except ValueError as e:
        print(f"⚠️  WARNING: Marching cubes failed with step_size=1, trying step_size=2")
        print(f"  Error: {e}")
        verts, faces, normals, values = skimage.measure.marching_cubes(
            sigmas_padded, level=level, spacing=[voxel_size] * 3,
            step_size=2
        )
        verts -= pad_width * voxel_size
    
    # Ensure consistent face orientation
    # Check if most normals point outward by sampling some face normals
    sample_indices = np.random.choice(len(faces), min(100, len(faces)), replace=False)
    outward_count = 0
    
    for idx in sample_indices:
        face = faces[idx]
        # Compute face center
        center = verts[face].mean(axis=0)
        # Compute face normal
        v0 = verts[face[1]] - verts[face[0]]
        v1 = verts[face[2]] - verts[face[0]]
        normal = np.cross(v0, v1)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        # Check if normal points outward (away from origin)
        if np.dot(normal, center) > 0:
            outward_count += 1
    
    # If most normals point inward, flip all faces
    if outward_count < len(sample_indices) / 2:
        print("Flipping face orientation for correct normals...")
        faces = faces[:, [0, 2, 1]]
    
    # Transform vertices to world coordinates
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
    
    # For each vertex, compute its score based on viewing angle evaluations
    vertex_scores = np.zeros(len(mesh_points))
    
    print("Computing vertex colors based on pose evaluations...")
    
    # For each vertex, find which camera angles would see it best
    for i, vertex in enumerate(tqdm(mesh_points, desc="Coloring vertices")):
        # Compute the angle from origin to vertex
        vertex_dir = vertex / (np.linalg.norm(vertex) + 1e-8)
        
        # Convert to spherical coordinates
        vertex_yaw = np.arctan2(vertex_dir[2], vertex_dir[0]) + np.pi
        vertex_pitch = np.arccos(np.clip(vertex_dir[1], -1, 1))
        
        # Find evaluations from similar viewing angles
        # Weight scores by angular distance
        total_weight = 0
        weighted_score = 0
        
        for eval_data in evaluations:
            # Angular distance between vertex normal and camera direction
            angle_dist = np.arccos(np.clip(
                np.cos(eval_data['yaw'] - vertex_yaw) * 
                np.sin(eval_data['pitch']) * np.sin(vertex_pitch) +
                np.cos(eval_data['pitch']) * np.cos(vertex_pitch), 
                -1, 1
            ))
            
            # Weight decreases with angular distance
            weight = np.exp(-angle_dist**2 / 0.5)  # Gaussian weighting
            
            weighted_score += eval_data['score'] * weight
            total_weight += weight
        
        vertex_scores[i] = weighted_score / (total_weight + 1e-8)
    
    # Normalize scores to [0, 1]
    vertex_scores = (vertex_scores - vertex_scores.min()) / (vertex_scores.max() - vertex_scores.min() + 1e-8)
    
    # Convert scores to colors using colormap
    # Use the new matplotlib API to avoid deprecation warning
    try:
        cmap = cm.colormaps[colormap]
    except AttributeError:
        # Fallback for older matplotlib versions
        cmap = cm.get_cmap(colormap)
    vertex_colors = cmap(vertex_scores)[:, :3]  # RGB only, no alpha
    
    # Post-process mesh to ensure quality
    if TRIMESH_AVAILABLE:
        print("Post-processing mesh for better quality...")
        import trimesh
        
        # Create trimesh object for processing
        temp_mesh = trimesh.Trimesh(vertices=mesh_points, faces=faces)
        
        # Initial mesh diagnostics
        initial_watertight = temp_mesh.is_watertight
        initial_volume = temp_mesh.volume if temp_mesh.is_volume else 0
        initial_faces = len(temp_mesh.faces)
        initial_vertices = len(temp_mesh.vertices)
        
        print("\n" + "="*50)
        print("Initial mesh status:")
        print(f"  Watertight: {initial_watertight}")
        print(f"  Volume: {initial_volume:.6f}")
        print(f"  Vertices: {initial_vertices}")
        print(f"  Faces: {initial_faces}")
        print("="*50 + "\n")
        
        # Step 1: Remove duplicate and degenerate geometry
        # Use new trimesh API to avoid deprecation warnings
        temp_mesh.update_faces(temp_mesh.unique_faces())
        temp_mesh.update_faces(temp_mesh.nondegenerate_faces())
        temp_mesh.remove_unreferenced_vertices()
        
        # Step 2: Fix normals to ensure consistency
        temp_mesh.fix_normals()
        
        # Step 3: Check for and fix various mesh issues
        if not temp_mesh.is_watertight:
            print("⚠️  WARNING: Mesh is not watertight!")
            print("Attempting comprehensive mesh repair...")
            
            # Try to fill holes
            try:
                holes_before = len(temp_mesh.facets_boundary)
                temp_mesh.fill_holes()
                holes_after = len(temp_mesh.facets_boundary)
                if holes_before > holes_after:
                    print(f"  ✓ Filled {holes_before - holes_after} holes")
            except Exception as e:
                print(f"  ✗ Hole filling failed: {e}")
            
            # Remove small disconnected components
            try:
                components = temp_mesh.split(only_watertight=False)
                if len(components) > 1:
                    # Keep only the largest component
                    largest = max(components, key=lambda m: len(m.vertices))
                    temp_mesh = largest
                    print(f"  ✓ Removed {len(components)-1} disconnected components")
            except Exception as e:
                print(f"  ✗ Component splitting failed: {e}")
            
            # Try to make manifold
            try:
                if not temp_mesh.is_winding_consistent:
                    temp_mesh.fix_normals()
                    print("  ✓ Fixed winding consistency")
            except Exception as e:
                print(f"  ✗ Winding fix failed: {e}")
        
        # Step 4: Check if mesh is inside-out
        if temp_mesh.is_watertight and temp_mesh.volume < 0:
            print("⚠️  WARNING: Mesh appears to be inside-out (negative volume)")
            temp_mesh.invert()
            print("  ✓ Inverted mesh orientation")
        
        # Step 5: Final validation
        final_watertight = temp_mesh.is_watertight
        final_volume = temp_mesh.volume if temp_mesh.is_volume else 0
        final_faces = len(temp_mesh.faces)
        final_vertices = len(temp_mesh.vertices)
        
        print("\n" + "="*50)
        print("Final mesh status:")
        print(f"  Watertight: {final_watertight}")
        print(f"  Volume: {final_volume:.6f}")
        print(f"  Vertices: {final_vertices}")
        print(f"  Faces: {final_faces}")
        print("="*50)
        
        # Alert if there are still issues
        if not final_watertight:
            print("\n⚠️  ALERT: Mesh still has issues after repair!")
            print("  - The mesh may appear hollow or have rendering artifacts")
            print("  - Consider increasing voxel resolution with --voxel_res")
            print("  - Or adjusting the isosurface level in the code")
        elif final_volume <= 0:
            print("\n⚠️  ALERT: Mesh has zero or negative volume!")
            print("  - The mesh may be degenerate or inside-out")
        else:
            print("\n✓ Mesh successfully processed and validated")
        
        # Update vertices and faces
        old_to_new = {}
        new_mesh_points = np.array(temp_mesh.vertices)
        
        # Map old vertices to new vertices
        print("Mapping vertices after post-processing...")
        for i, old_v in enumerate(tqdm(mesh_points, desc="Mapping vertices")):
            # Find closest new vertex
            distances = np.linalg.norm(new_mesh_points - old_v, axis=1)
            old_to_new[i] = np.argmin(distances)
        
        # Update vertex colors for new vertices
        new_vertex_scores = np.zeros(len(new_mesh_points))
        for old_idx, new_idx in old_to_new.items():
            if old_idx < len(vertex_scores):
                new_vertex_scores[new_idx] = vertex_scores[old_idx]
        
        # Recompute colors
        vertex_scores = new_vertex_scores
        # Use the same colormap instance we already have
        vertex_colors = cmap(vertex_scores)[:, :3]
        
        mesh_points = new_mesh_points
        faces = np.array(temp_mesh.faces)
        
        print(f"Post-processing complete. Watertight: {temp_mesh.is_watertight}")
    
    # Create PLY with colors
    num_verts = len(mesh_points)
    num_faces = len(faces)
    
    # Define vertex data with colors
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_data = np.zeros(num_verts, dtype=vertex_dtype)
    
    print("Creating PLY vertex data...")
    for i in tqdm(range(num_verts), desc="Preparing vertices"):
        vertex_data[i] = (
            mesh_points[i, 0], mesh_points[i, 1], mesh_points[i, 2],
            int(vertex_colors[i, 0] * 255),
            int(vertex_colors[i, 1] * 255),
            int(vertex_colors[i, 2] * 255)
        )
    
    # Define face data
    face_dtype = [('vertex_indices', 'i4', (3,))]
    face_data = np.array([(face,) for face in faces], dtype=face_dtype)
    
    # Create PLY elements
    vertex_el = PlyElement.describe(vertex_data, 'vertex')
    face_el = PlyElement.describe(face_data, 'face')
    
    # Write PLY file
    print("Writing PLY file...")
    ply_data = PlyData([vertex_el, face_el])
    ply_data.write(output_path)
    print(f"Saved colored heatmap mesh to {output_path}")
    
    # Analyze mesh quality
    print(f"Mesh statistics:")
    print(f"  Vertices: {num_verts}")
    print(f"  Faces: {num_faces}")
    print(f"  Bounds: X[{mesh_points[:, 0].min():.3f}, {mesh_points[:, 0].max():.3f}], "
          f"Y[{mesh_points[:, 1].min():.3f}, {mesh_points[:, 1].max():.3f}], "
          f"Z[{mesh_points[:, 2].min():.3f}, {mesh_points[:, 2].max():.3f}]")
    
    return vertex_scores




def render_ply_video_open3d(ply_path, output_video_path, num_frames=90, cfg='Head', resolution=(512, 512)):
    """
    Render a video of the PLY mesh using Open3D for high-quality visualization.
    """
    print(f"Rendering PLY video from {ply_path} using Open3D...")
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(ply_path)
    
    # Ensure mesh has vertex normals for proper shading
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Check if mesh is watertight
    is_watertight = mesh.is_watertight()
    print(f"Mesh is watertight: {is_watertight}")
    
    if not is_watertight:
        print("Attempting to repair mesh...")
        # Remove duplicate vertices and faces
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        
        # Try to make it watertight
        mesh.compute_vertex_normals()
        
    # Get mesh bounds for camera setup
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    max_extent = np.max(extent)
    
    # Camera parameters
    camera_lookat_point = np.array([0, 0, 0.2]) if cfg == 'FFHQ' else np.array([0, 0, 0])
    radius = 2.7
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=resolution[0], height=resolution[1])
    vis.add_geometry(mesh)
    
    # Set rendering options for better quality
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background
    opt.mesh_show_wireframe = False
    opt.mesh_show_back_face = True  # Show back faces to avoid hollow appearance
    opt.light_on = True
    
    # Get view control
    ctr = vis.get_view_control()
    
    # Create video writer
    writer = imageio.get_writer(output_video_path, fps=30)
    frames = []
    
    for frame_idx in tqdm(range(num_frames), desc="Rendering mesh video with Open3D"):
        # Camera motion - same as in the original video generation
        if cfg == "Head":
            pitch_range = 0.5
            yaw = np.pi/2 + 2 * np.pi * frame_idx / num_frames
            pitch = np.pi/2 - 0.05 + pitch_range * np.sin(2 * np.pi * frame_idx / num_frames)
        else:
            pitch_range = 0.25
            yaw_range = 1.5
            yaw = np.pi/2 + yaw_range * np.sin(2 * np.pi * frame_idx / num_frames)
            pitch = np.pi/2 - 0.05 + pitch_range * np.cos(2 * np.pi * frame_idx / num_frames)
        
        # Convert spherical to Cartesian coordinates
        cam_x = radius * np.sin(pitch) * np.cos(yaw)
        cam_y = radius * np.cos(pitch)
        cam_z = radius * np.sin(pitch) * np.sin(yaw)
        cam_pos = np.array([cam_x, cam_y, cam_z]) + camera_lookat_point
        
        # Set camera parameters
        ctr.set_lookat(camera_lookat_point)
        ctr.set_up([0, 1, 0])
        ctr.set_front(camera_lookat_point - cam_pos)
        ctr.set_zoom(0.7)
        
        # Update geometry (in case of color changes)
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        img = vis.capture_screen_float_buffer(do_render=True)
        img = np.asarray(img)
        img = (img * 255).astype(np.uint8)
        
        frames.append(img)
        writer.append_data(img)
    
    writer.close()
    vis.destroy_window()
    print(f"Saved mesh video to {output_video_path}")
    
    # Save grid of frames
    if len(frames) > 0:
        indices = np.linspace(0, len(frames)-1, 16, dtype=int)
        selected_frames = [frames[i] for i in indices]
        
        grid = np.zeros((4 * selected_frames[0].shape[0], 
                        4 * selected_frames[0].shape[1], 3), dtype=np.uint8)
        
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(selected_frames):
                    grid[i*selected_frames[0].shape[0]:(i+1)*selected_frames[0].shape[0],
                         j*selected_frames[0].shape[1]:(j+1)*selected_frames[0].shape[1]] = selected_frames[idx]
        
        grid_path = output_video_path.replace('.mp4', '_grid.png')
        imageio.imwrite(grid_path, grid)
        print(f"Saved mesh video grid to {grid_path}")


def render_ply_video(ply_path, output_video_path, num_frames=90, cfg='Head', resolution=(512, 512)):
    """
    Render a video of the PLY mesh rotating with the same camera trajectory as the original video.
    """
    # Try Open3D first (best quality), then pyrender
    if OPEN3D_AVAILABLE:
        try:
            print("Using Open3D for high-quality mesh rendering...")
            render_ply_video_open3d(ply_path, output_video_path, num_frames, cfg, resolution)
            return
        except Exception as e:
            print(f"Open3D rendering failed: {e}")
            print("Falling back to pyrender...")
    
    if not PYRENDER_AVAILABLE:
        raise ImportError("Neither Open3D nor pyrender is available. Please install one: pip install open3d OR pip install pyrender")
    render_ply_video_pyrender(ply_path, output_video_path, num_frames, cfg, resolution)


def render_ply_video_pyrender(ply_path, output_video_path, num_frames=90, cfg='Head', resolution=(512, 512)):
    """
    Render a video of the PLY mesh using pyrender.
    """
    if not PYRENDER_AVAILABLE:
        raise ImportError("pyrender is not available")
        
    print(f"Rendering PLY video from {ply_path} using pyrender...")
    
    # Set up headless rendering for remote servers
    # Try different platform options
    platforms = ['egl', 'osmesa']
    renderer = None
    
    for platform in platforms:
        try:
            os.environ['PYOPENGL_PLATFORM'] = platform
            print(f"Trying pyrender with {platform.upper()} backend...")
            
            # Test renderer creation
            test_renderer = pyrender.OffscreenRenderer(10, 10)
            test_renderer.delete()
            
            print(f"Successfully initialized pyrender with {platform.upper()}")
            break
        except Exception as e:
            print(f"Failed to initialize pyrender with {platform}: {e}")
            if platform == platforms[-1]:
                raise RuntimeError("Could not initialize pyrender with any backend. Please use --force_matplotlib flag.")
    
    # Load the mesh with proper vertex color handling
    mesh = trimesh.load(ply_path)
    
    # Debug: Check mesh properties
    print(f"Mesh vertices: {len(mesh.vertices)}")
    print(f"Mesh faces: {len(mesh.faces)}")
    
    # Try to extract vertex colors directly from the PLY file if trimesh fails
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    
    # Extract vertex colors from PLY
    vertex_data = plydata['vertex']
    vertex_names = vertex_data.data.dtype.names
    vertex_colors = None
    if vertex_names and 'red' in vertex_names:
        vertex_colors = np.vstack([
            vertex_data['red'],
            vertex_data['green'],
            vertex_data['blue']
        ]).T.astype(np.float32) / 255.0
        print(f"Loaded vertex colors from PLY: shape={vertex_colors.shape}, range=[{vertex_colors.min():.3f}, {vertex_colors.max():.3f}]")
        
        # Apply colors to trimesh mesh
        mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
    
    # Ensure mesh has proper normals
    if not mesh.vertex_normals.any():
        mesh.compute_vertex_normals()
    
    # Create pyrender mesh
    if vertex_colors is not None:
        # Create pyrender mesh with vertex colors
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        print("Created pyrender mesh with vertex colors")
    else:
        # No vertex colors, use default material
        print("No vertex colors found, using default material")
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.7, 0.7, 0.7, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5,
            alphaMode='OPAQUE',
            doubleSided=False
        )
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    
    # Create scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0])
    scene.add(mesh_pyrender)
    
    # Add lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose = np.eye(4)
    light_pose[:3, 2] = [0.3, -0.5, -1.0]  # Light direction
    scene.add(light, pose=light_pose)
    
    # Add another light from opposite direction for better illumination
    light2 = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=1.0)
    light_pose2 = np.eye(4)
    light_pose2[:3, 2] = [-0.3, 0.2, 1.0]
    scene.add(light2, pose=light_pose2)
    
    # Create renderer
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    
    # Camera parameters - match the original video
    camera_lookat_point = np.array([0, 0, 0.2]) if cfg == 'FFHQ' else np.array([0, 0, 0])
    radius = 2.7
    
    # Create video writer
    writer = imageio.get_writer(output_video_path, fps=30)
    
    frames = []
    for frame_idx in tqdm(range(num_frames), desc="Rendering mesh video"):
        # Camera motion - same as in the original video generation
        if cfg == "Head":
            pitch_range = 0.5
            yaw = np.pi/2 + 2 * np.pi * frame_idx / num_frames
            pitch = np.pi/2 - 0.05 + pitch_range * np.sin(2 * np.pi * frame_idx / num_frames)
        else:
            pitch_range = 0.25
            yaw_range = 1.5
            yaw = np.pi/2 + yaw_range * np.sin(2 * np.pi * frame_idx / num_frames)
            pitch = np.pi/2 - 0.05 + pitch_range * np.cos(2 * np.pi * frame_idx / num_frames)
        
        # Convert spherical to Cartesian coordinates
        cam_x = radius * np.sin(pitch) * np.cos(yaw)
        cam_y = radius * np.cos(pitch)
        cam_z = radius * np.sin(pitch) * np.sin(yaw)
        cam_pos = np.array([cam_x, cam_y, cam_z]) + camera_lookat_point
        
        # Create camera pose matrix
        # Look at the center
        forward = camera_lookat_point - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Compute right vector
        up_temp = np.array([0, 1, 0])
        right = np.cross(forward, up_temp)
        right = right / np.linalg.norm(right)
        
        # Compute up vector
        up = np.cross(right, forward)
        
        # Build camera matrix
        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward  # OpenGL convention: camera looks down -Z
        camera_pose[:3, 3] = cam_pos
        
        # Create camera
        camera = pyrender.PerspectiveCamera(yfov=np.radians(18.837))
        cam_node = scene.add(camera, pose=camera_pose)
        
        # Render
        color, _ = renderer.render(scene)
        
        # Remove camera for next frame
        scene.remove_node(cam_node)
        
        # Save frame
        # Ensure color is in uint8 format
        if color.dtype != np.uint8:
            if color.max() <= 1.0:
                color = (color * 255).astype(np.uint8)
            else:
                color = color.astype(np.uint8)
        frames.append(color)
        writer.append_data(color)
    
    writer.close()
    renderer.delete()
    print(f"Saved mesh video to {output_video_path}")
    
    # Save grid of frames
    if len(frames) > 0:
        indices = np.linspace(0, len(frames)-1, 16, dtype=int)
        selected_frames = [frames[i] for i in indices]
        
        grid = np.zeros((4 * selected_frames[0].shape[0], 
                        4 * selected_frames[0].shape[1], 3), dtype=np.uint8)
        
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(selected_frames):
                    grid[i*selected_frames[0].shape[0]:(i+1)*selected_frames[0].shape[0],
                         j*selected_frames[0].shape[1]:(j+1)*selected_frames[0].shape[1]] = selected_frames[idx]
        
        grid_path = output_video_path.replace('.mp4', '_grid.png')
        imageio.imwrite(grid_path, grid)
        print(f"Saved mesh video grid to {grid_path}")


def create_comparison_video(rgb_path, depth_path, mesh_path, evaluations, output_path, 
                          num_frames=90, cfg='Head', fps=30):
    """
    Create a side-by-side video with RGB, depth, mesh, and animated performance plot.
    """
    from PIL import Image
    print("Loading video files...")
    
    # Load videos
    rgb_reader = imageio.get_reader(rgb_path)
    mesh_reader = imageio.get_reader(mesh_path)
    depth_reader = imageio.get_reader(depth_path) if depth_path else None
    
    # Get frame dimensions
    rgb_frame = rgb_reader.get_data(0)
    mesh_frame = mesh_reader.get_data(0)
    h, w = rgb_frame.shape[:2]
    
    # Target size for each panel (scale up small videos)
    target_size = 512  # Fixed size for each panel
    if h < target_size or w < target_size:
        # Scale up to target size
        panel_h = target_size
        panel_w = target_size
    else:
        panel_h = h
        panel_w = w
    
    # Setup plot dimensions
    plot_w = panel_w
    plot_h = panel_h
    
    # Create output layout: 1x4 single row
    # [RGB | Depth | Mesh | Plot]
    out_w = panel_w * 4
    out_h = panel_h
    
    # Create video writer with fixed dimensions
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', 
                               output_params=['-s', f'{out_w}x{out_h}'])
    
    # Extract scores and angles from evaluations, sorted by frame index
    frame_indices = [e['frame_idx'] for e in evaluations]
    sorted_indices = np.argsort(frame_indices)
    
    # Create lists indexed by frame
    all_scores = []
    all_angles = []
    for i in sorted_indices:
        eval_data = evaluations[i]
        yaw = eval_data['yaw']
        # Convert to degrees for display
        angle = (yaw * 180 / np.pi) % 360
        all_angles.append(angle)
        all_scores.append(eval_data['score'])
    
    print(f"Creating {num_frames} frames for comparison video...")
    
    for frame_idx in tqdm(range(num_frames), desc="Creating comparison frames"):
        # Read frames
        try:
            rgb_frame = rgb_reader.get_data(frame_idx)
            mesh_frame = mesh_reader.get_data(frame_idx)
            depth_frame = depth_reader.get_data(frame_idx) if depth_reader else None
        except:
            # Handle end of video
            break
        
        # Create output frame
        output_frame = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
        
        # Resize frames if needed
        def resize_frame(frame, target_h, target_w):
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                return np.array(pil_img)
            return frame
        
        # Resize all frames to panel size
        rgb_frame_resized = resize_frame(rgb_frame[:, :, :3], panel_h, panel_w)
        # Fix horizontal flip in mesh by flipping it back
        mesh_frame_flipped = mesh_frame[:, ::-1, :3]  # Flip horizontally
        mesh_frame_resized = resize_frame(mesh_frame_flipped, panel_h, panel_w)
        depth_frame_resized = resize_frame(depth_frame[:, :, :3], panel_h, panel_w) if depth_frame is not None else None
        
        # Place frames in single row
        if depth_frame_resized is not None:
            # [RGB | Depth | Mesh | Plot]
            output_frame[:, :panel_w] = rgb_frame_resized  # RGB
            output_frame[:, panel_w:2*panel_w] = depth_frame_resized  # Depth
            output_frame[:, 2*panel_w:3*panel_w] = mesh_frame_resized  # Mesh
        else:
            # [RGB | Mesh | (empty) | Plot]
            output_frame[:, :panel_w] = rgb_frame_resized  # RGB
            output_frame[:, panel_w:2*panel_w] = mesh_frame_resized  # Mesh
            output_frame[:, 2*panel_w:3*panel_w] = 255  # Empty white
        
        # Create animated plot with fixed size
        fig = plt.figure(figsize=(plot_w/100, plot_h/100), dpi=100)
        ax = fig.add_subplot(111)
        
        # Set fixed aspect ratio and position with better margins
        # [left, bottom, width, height] - increase margins for labels
        ax.set_position([0.15, 0.12, 0.78, 0.78])  # More space for labels
        
        # Get current values
        if frame_idx < len(all_scores):
            current_score = all_scores[frame_idx]
            current_angle = all_angles[frame_idx]
        else:
            # If we have more frames than evaluations, use the last values
            current_score = all_scores[-1] if all_scores else 0
            current_angle = all_angles[-1] if all_angles else 0
        
        # Plot all scores up to current frame
        frames_to_plot = min(frame_idx + 1, len(all_scores))
        if frames_to_plot > 0:
            # X-axis is frame index (time)
            x_values = list(range(frames_to_plot))
            y_values = all_scores[:frames_to_plot]
            
            # Plot trajectory line with improved styling
            if len(x_values) > 1:
                ax.plot(x_values, y_values, 
                        'b-', linewidth=2.5, alpha=0.9, label='Performance')
            
            # Plot all points with better visibility
            ax.scatter(x_values, y_values, 
                      c='blue', s=40, alpha=0.7, zorder=5, edgecolors='darkblue', linewidth=0.5)
            
            # Highlight current position
            if frame_idx < len(all_scores):
                ax.plot(frame_idx, current_score, 'ro', markersize=16, zorder=10, 
                       markeredgecolor='darkred', markeredgewidth=1.5)
                ax.axvline(x=frame_idx, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        
        # Set x-axis limits with sliding window
        window_width = min(120, num_frames)  # Show up to 120 frames at a time, but not more than total frames
        
        if num_frames <= window_width:
            # If total frames fit in window, show all
            x_min = 0
            x_max = num_frames
        elif frame_idx < window_width - 20:
            # At the beginning, show fixed window
            x_min = 0
            x_max = window_width
        else:
            # Slide the window to keep current position visible
            x_min = max(0, frame_idx - window_width + 20)  # Keep some history visible
            x_max = min(num_frames, frame_idx + 20)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.05, 1.15)  # Slight padding for y-axis
        
        # Labels with adjusted font sizes
        ax.set_xlabel('Frame', fontsize=12, labelpad=5)
        ax.set_ylabel('Neural Network Score', fontsize=12, labelpad=10)
        # Title removed per user request
        
        # Improved grid styling
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)  # Grid behind data
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Add text annotations
        if frame_idx < len(all_scores):
            # Score annotation - move slightly inward from edge
            ax.text(0.97, 0.95, f'Score: {current_score:.3f}', 
                    transform=ax.transAxes, fontsize=11, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
            
            # Angle annotation - move slightly inward from edge
            ax.text(0.03, 0.95, f'Angle: {current_angle:.0f}°', 
                    transform=ax.transAxes, fontsize=11, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # Frame annotation - move slightly up from bottom
            ax.text(0.5, 0.04, f'Frame {frame_idx}/{num_frames}', 
                    transform=ax.transAxes, fontsize=9, va='bottom', ha='center',
                    color='gray', alpha=0.7)
        
        # Convert plot to image without tight_layout (which can change size)
        fig.canvas.draw()
        # Get the buffer in RGBA format and convert to RGB
        buf = fig.canvas.buffer_rgba()
        buf_array = np.asarray(buf)
        # Convert RGBA to RGB
        plot_frame = buf_array[:, :, :3]
        plt.close(fig)
        
        # The plot should already be the correct size, but ensure it matches
        if plot_frame.shape[0] != plot_h or plot_frame.shape[1] != plot_w:
            # Force resize to exact dimensions
            from PIL import Image
            plot_img = Image.fromarray(plot_frame)
            plot_img = plot_img.resize((plot_w, plot_h), Image.Resampling.LANCZOS)
            plot_frame = np.array(plot_img)
        
        # Place plot in 4th position (rightmost)
        output_frame[:, 3*panel_w:4*panel_w] = plot_frame
        
        # Add labels
        output_frame = add_labels(output_frame, panel_w, panel_h, has_depth=depth_frame is not None)
        
        # Ensure output frame is exactly the expected size
        if output_frame.shape != (out_h, out_w, 3):
            print(f"Warning: Frame size mismatch. Expected {(out_h, out_w, 3)}, got {output_frame.shape}")
            # Force resize if needed
            from PIL import Image
            img = Image.fromarray(output_frame)
            img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
            output_frame = np.array(img)
        
        # Write frame
        writer.append_data(output_frame)
    
    # Cleanup
    writer.close()
    rgb_reader.close()
    mesh_reader.close()
    if depth_reader:
        depth_reader.close()
    
    print(f"Comparison video saved to {output_path}")


def add_labels(frame, w, h, has_depth=True):
    """Add text labels to each panel of the comparison video."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Convert to PIL Image
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Define labels for single row layout
    if has_depth:
        labels = ['RGB', 'Depth', 'Heatmap Mesh', 'Performance Over Time']
        positions = [(10, 10), (w + 10, 10), (2*w + 10, 10), (3*w + 10, 10)]
    else:
        labels = ['RGB', 'Heatmap Mesh', '', 'Performance Over Time']
        positions = [(10, 10), (w + 10, 10), (2*w + 10, 10), (3*w + 10, 10)]
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
    
    for label, pos in zip(labels, positions):
        if label:  # Skip empty labels
            x, y = pos
            # Get text bbox
            bbox = draw.textbbox((x, y), label, font=font)
            # Draw white background
            draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill='white')
            # Draw text
            draw.text((x, y), label, fill='black', font=font)
    
    return np.array(img)


def generate_consistent_identity_video(G, output_path, evaluator, seed=0, num_frames=90, 
                                     cfg='Head', device=torch.device('cuda'),
                                     white_back=True, nrr=64, save_depth=True,
                                     generate_heatmap=True, voxel_res=128,
                                     truncation_psi=0.6, sample_mult=1.0, colormap='viridis'):
    """Generate video with consistent identity throughout rotation."""
    
    print(f"Generating video with consistent identity (seed={seed})...")
    
    # Generate latent code ONCE for consistent identity
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    
    # Camera parameters
    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    
    # Generate initial conditioning for mapping
    # This is crucial - we need to use the same conditioning for mapping to maintain identity
    pose_cond_rad = 90/180*np.pi  # Default 90 degrees
    initial_cam2world = LookAtPoseSampler.sample(pose_cond_rad, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    initial_c = torch.cat([initial_cam2world.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    # Generate w vector ONCE with initial conditioning - this ensures consistent identity
    w = G.mapping(z=z, c=initial_c, truncation_psi=truncation_psi, truncation_cutoff=14)
    
    # Initialize evaluator
    evaluator.initialize({
        'cfg': cfg,
        'device': device,
        'seed': seed
    })
    
    # Video writers
    writer = imageio.get_writer(output_path, fps=30)
    depth_writer = None
    if save_depth:
        depth_path = output_path.replace('.mp4', '_depth.mp4')
        depth_writer = imageio.get_writer(depth_path, fps=30)
    
    frames = []
    depth_frames = []
    evaluations = []  # Store pose evaluations for heatmap
    
    for frame_idx in tqdm(range(num_frames), desc="Generating frames"):
        # Camera motion - only change the camera pose, not the identity
        if cfg == "Head":
            pitch_range = 0.5
            yaw = 3.14/2 + 2 * 3.14 * frame_idx / num_frames
            pitch = 3.14/2 - 0.05 + pitch_range * np.sin(2 * 3.14 * frame_idx / num_frames)
        else:
            pitch_range = 0.25
            yaw_range = 1.5
            yaw = 3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / num_frames)
            pitch = 3.14/2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / num_frames)
        
        # Generate camera pose for this frame
        cam2world_pose = LookAtPoseSampler.sample(yaw, pitch, camera_lookat_point, radius=2.7, device=device)
        
        # Create conditioning vector for synthesis (NOT for mapping!)
        c_synth = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        
        # Generate image using the SAME w vector but different camera pose
        synthesis_result = G.synthesis(ws=w, c=c_synth, noise_mode='const')
        img = synthesis_result['image'][0]
        
        # Extract depth if available
        depth_img = None
        if save_depth and 'image_depth' in synthesis_result:
            depth = synthesis_result['image_depth'][0]
            # Normalize depth for visualization
            depth = -depth  # Invert so closer is brighter
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = torch.zeros_like(depth)
            
            # Convert to RGB for video (grayscale)
            depth_img = depth.clamp(0, 1).cpu().numpy()
            if len(depth_img.shape) == 2:
                depth_img = np.expand_dims(depth_img, 0)
            # Make it 3-channel for video
            depth_img = np.repeat(depth_img, 3, axis=0)
            depth_img = np.transpose(depth_img, (1, 2, 0))
            depth_img = (depth_img * 255).astype(np.uint8)
            depth_frames.append(depth_img)
        
        # Handle white background if requested
        if white_back and 'image_mask' in synthesis_result:
            mask = synthesis_result['image_mask'][0]
            # Resize mask if needed
            if mask.shape[-2:] != img.shape[-2:]:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0), 
                    size=img.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            if mask.shape[0] == 1:
                mask = mask.repeat(3, 1, 1)
            # Composite with white background
            white_bg = torch.ones_like(img)
            img = img * mask + white_bg * (1 - mask)
        
        # Convert to RGB
        img_np = img.clamp(-1, 1).cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np + 1) / 2 * 255
        img_np = img_np.astype(np.uint8)
        
        frames.append(img_np)
        writer.append_data(img_np)
        
        if depth_writer is not None and depth_img is not None:
            depth_writer.append_data(depth_img)
        
        # Evaluate pose for heatmap
        if generate_heatmap:
            # Prepare sample data for evaluator
            sample_data = {
                'image': img_np,
                'depth': depth_img,
                'pose': {'yaw': yaw, 'pitch': pitch},
                'seed': seed,
                'frame_idx': frame_idx
            }
            
            score = evaluator.evaluate(sample_data)
            evaluations.append({
                'yaw': yaw,
                'pitch': pitch,
                'score': score,
                'frame_idx': frame_idx
            })
    
    writer.close()
    print(f"Saved video to {output_path}")
    
    if depth_writer is not None:
        depth_writer.close()
        print(f"Saved depth video to {depth_path}")
    
    # Finalize evaluator
    evaluator.finalize(evaluations)
    
    # Save a grid of frames for comparison
    if len(frames) > 0:
        indices = np.linspace(0, len(frames)-1, 16, dtype=int)
        selected_frames = [frames[i] for i in indices]
        
        grid = np.zeros((4 * selected_frames[0].shape[0], 
                        4 * selected_frames[0].shape[1], 3), dtype=np.uint8)
        
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(selected_frames):
                    grid[i*selected_frames[0].shape[0]:(i+1)*selected_frames[0].shape[0],
                         j*selected_frames[0].shape[1]:(j+1)*selected_frames[0].shape[1]] = selected_frames[idx]
        
        grid_path = output_path.replace('.mp4', '_grid.png')
        imageio.imwrite(grid_path, grid)
        print(f"Saved frame grid to {grid_path}")
        
        # Save depth grid if available
        if len(depth_frames) > 0:
            selected_depth_frames = [depth_frames[i] for i in indices]
            
            depth_grid = np.zeros((4 * selected_depth_frames[0].shape[0], 
                                 4 * selected_depth_frames[0].shape[1], 3), dtype=np.uint8)
            
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    if idx < len(selected_depth_frames):
                        depth_grid[i*selected_depth_frames[0].shape[0]:(i+1)*selected_depth_frames[0].shape[0],
                                  j*selected_depth_frames[0].shape[1]:(j+1)*selected_depth_frames[0].shape[1]] = selected_depth_frames[idx]
            
            depth_grid_path = output_path.replace('.mp4', '_depth_grid.png')
            imageio.imwrite(depth_grid_path, depth_grid)
            print(f"Saved depth frame grid to {depth_grid_path}")
    
    # Generate heatmap mesh if requested
    if generate_heatmap and len(evaluations) > 0:
        print("\nGenerating 3D shape and heatmap mesh...")
        
        # Extract 3D shape using SDF
        samples, voxel_origin, voxel_size = create_samples(
            N=voxel_res,
            voxel_origin=[0, 0, 0],
            cube_length=G.rendering_kwargs['box_warp']
        )
        samples = samples.to(device)
        
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], samples.shape[1], 3), device=device)
        transformed_ray_directions_expanded[..., -1] = -1
        
        # Extract SDF in batches
        max_batch = 500000 if device.type == 'cuda' else 50000
        head = 0
        
        with torch.no_grad():
            pbar = tqdm(total=samples.shape[1], desc="Extracting SDF")
            while head < samples.shape[1]:
                batch_size = min(max_batch, samples.shape[1] - head)
                sigma = G.sample_mixed(
                    samples[:, head:head+batch_size],
                    transformed_ray_directions_expanded[:, :batch_size],
                    w,
                    truncation_psi=truncation_psi,
                    noise_mode='const'
                )['sigma']
                sigmas[:, head:head+batch_size] = sigma
                head += batch_size
                pbar.update(batch_size)
            pbar.close()
        
        sigmas = sigmas.reshape((voxel_res, voxel_res, voxel_res)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)
        
        # Apply padding to remove artifacts
        pad = int(30 * voxel_res / 256)
        pad_top = int(38 * voxel_res / 256)
        sigmas[:pad] = 0
        sigmas[-pad:] = 0
        sigmas[:, :pad] = 0
        sigmas[:, -pad_top:] = 0
        sigmas[:, :, :pad] = 0
        sigmas[:, :, -pad:] = 0
        
        # Additional cleanup: remove isolated high values (noise)
        # These can create small floating artifacts
        from scipy import ndimage
        # Create binary mask of significant values
        mask = sigmas > 5.0  # Values above threshold
        # Remove small connected components
        labeled, num_features = ndimage.label(mask)
        if num_features > 1:
            # Find the largest component
            component_sizes = np.array([np.sum(labeled == i) for i in range(1, num_features + 1)])
            largest_component = np.argmax(component_sizes) + 1
            # Keep only the largest component
            sigmas[labeled != largest_component] = 0
            print(f"Removed {num_features - 1} small disconnected components from volume")
        
        # Create colored mesh
        mesh_path = output_path.replace('.mp4', '_heatmap_mesh.ply')
        vertex_scores = create_colored_mesh_ply(sigmas, voxel_origin, voxel_size, evaluations, mesh_path, colormap)
        
        # Render video of the colored mesh
        mesh_video_path = output_path.replace('.mp4', '_heatmap_mesh_video.mp4')
        try:
            # Try rendering with best available method
            print("Rendering mesh video...")
            render_ply_video(mesh_path, mesh_video_path, num_frames=num_frames, cfg=cfg)
        except Exception as e:
            print(f"Warning: Could not render mesh video: {type(e).__name__}: {e}")
            print("Full traceback:")
            import traceback
            traceback.print_exc()
            print("The PLY file was still saved successfully.")
        
        # Create score distribution plot
        scores = [e['score'] for e in evaluations]
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Fake Neural Network Score')
        plt.ylabel('Frequency')
        plt.title(f'Score Distribution Across {num_frames} Poses (Seed {seed})')
        plt.grid(True, alpha=0.3)
        score_dist_path = output_path.replace('.mp4', '_score_distribution.png')
        plt.savefig(score_dist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved score distribution to {score_dist_path}")
        
        # Create side-by-side video with all components
        if os.path.exists(output_path) and os.path.exists(mesh_video_path):
            print("\nCreating side-by-side comparison video...")
            comparison_path = output_path.replace('.mp4', '_comparison.mp4')
            try:
                create_comparison_video(
                    output_path,  # Original RGB video
                    depth_path if save_depth and os.path.exists(depth_path) else None,
                    mesh_video_path,
                    evaluations,
                    comparison_path,
                    num_frames=num_frames,
                    cfg=cfg
                )
                print(f"Saved comparison video to {comparison_path}")
            except Exception as e:
                print(f"Warning: Could not create comparison video: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate video with consistent identity')
    parser.add_argument('--network', type=str, default='../models/spherehead-ckpt-025000.pkl',
                       help='Path to network pickle (default: ../models/spherehead-ckpt-025000.pkl)')
    parser.add_argument('--seeds', type=str, default='0', 
                       help='Random seeds (comma-separated or range, e.g., "0,1,2" or "0-2")')
    parser.add_argument('--outdir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--no_white_back', action='store_true', default=False,
                       help='Disable white background (default: white background enabled)')
    parser.add_argument('--cfg', type=str, default='Head', choices=['FFHQ', 'Head', 'Cats'],
                       help='Configuration preset (default: Head)')
    parser.add_argument('--nrr', type=int, default=64,
                       help='Neural rendering resolution override. Higher = better quality but slower. Common values: 64 (fast), 128 (high quality), 256 (maximum) (default: 64)')
    parser.add_argument('--num_frames', type=int, default=90,
                       help='Number of frames in the video (default: 90)')
    parser.add_argument('--trunc', '--truncation_psi', dest='truncation_psi', type=float, default=0.6,
                       help='Truncation psi: controls quality vs diversity trade-off. Lower values (0.5-0.8) produce higher quality, more typical results. 1.0 = full diversity but more artifacts (default: 0.6)')
    parser.add_argument('--sample_mult', type=float, default=1.0,
                       help='Multiplier for depth sampling in volume rendering. Higher = better depth accuracy but slower (default: 1.0)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto-selects best GPU if not specified)')
    parser.add_argument('--no_depth', action='store_true', default=False,
                       help='Disable depth map saving (default: depth maps enabled)')
    parser.add_argument('--no_heatmap', action='store_true', default=False,
                       help='Disable heatmap mesh generation (default: heatmap enabled)')
    parser.add_argument('--voxel_res', type=int, default=128,
                       help='Voxel resolution for 3D shape extraction. Higher = more detailed mesh but uses more memory. Common values: 128, 256, 512 (default: 128)')
    
    # Evaluator parameters
    parser.add_argument('--evaluator', type=str, default=None,
                       help='Path to custom evaluator Python file. Should contain a class '
                            'inheriting from BaseEvaluator (default: synth_eval/example_pose_evaluator.py)')
    parser.add_argument('--evaluator_args', nargs='*', default=[],
                       help='Arguments for custom evaluator as key=value pairs '
                            '(e.g., threshold=0.5 model_path=/path/to/model)')
    
    # Visualization parameters
    parser.add_argument('--colormap', type=str, default='viridis',
                       help='Colormap for heatmap visualization. Options: viridis, plasma, hot, coolwarm, turbo, magma, inferno, cividis, etc. (default: viridis)')
    
    args = parser.parse_args()
    
    # Parse seeds
    if '-' in args.seeds:
        start, end = map(int, args.seeds.split('-'))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(',')]
    
    # Parse evaluator arguments
    evaluator_kwargs = parse_key_value_pairs(args.evaluator_args) if args.evaluator_args else {}
    
    # Load or create evaluator
    if args.evaluator:
        print(f"Loading custom evaluator from {args.evaluator}")
        evaluator = load_custom_evaluator(args.evaluator, evaluator_kwargs)
    else:
        # Use the example external evaluator
        example_evaluator_path = os.path.join(os.path.dirname(__file__), 'example_pose_evaluator.py')
        if os.path.exists(example_evaluator_path):
            print("Using example pose-based evaluator from example_pose_evaluator.py")
            evaluator = load_custom_evaluator(example_evaluator_path, evaluator_kwargs)
        else:
            raise FileNotFoundError(
                f"Example evaluator not found at {example_evaluator_path}. "
                "Please specify an evaluator with --evaluator or ensure example_pose_evaluator.py exists."
            )
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.network):
        print(f"Error: Model not found at {args.network}")
        # Try to find the model in common locations
        possible_paths = [
            '../models/spherehead-ckpt-025000.pkl',
            '../../models/spherehead-ckpt-025000.pkl',
            os.path.join(os.path.dirname(__file__), '../models/spherehead-ckpt-025000.pkl'),
            'models/spherehead-ckpt-025000.pkl',
            'spherehead-ckpt-025000.pkl'
        ]
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found model at: {path}")
                args.network = path
                found = True
                break
        if not found:
            print("Could not find model in common locations.")
            print("Please specify the correct path with --network")
            print("Example: python pose_robustness/demo_minimal_memory.py --network path/to/model.pkl")
            sys.exit(1)
    
    # Set environment variable for memory
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Get device
    if args.device:
        device = torch.device(args.device)
        print(f'Using specified device: {device}')
    else:
        device = get_best_gpu_device()
    
    # Clear GPU memory
    clear_gpu_memory(device)
    
    print("="*60)
    print("SYNTHETIC EVALUATION TOOL")
    print("="*60)
    print(f"Model: {args.network}")
    print(f"Configuration preset: {args.cfg}")
    print(f"Seeds: {seeds}")
    print(f"Device: {device}")
    print(f"Output directory: {args.outdir}")
    print(f"Evaluator: {args.evaluator or 'example_pose_evaluator.py'}")
    print("="*60)
    
    # Load model
    print(f"Loading model from {args.network}...")
    with dnnlib.util.open_url(args.network) as f:
        data = load_network_pkl_cpu_safe(f, device)
        G = data['G_ema'].to(device)
    
    # Configure model
    G.rendering_kwargs['white_back'] = not args.no_white_back
    if args.nrr is not None:
        G.neural_rendering_resolution = args.nrr
    
    # Apply sample multiplier
    if args.sample_mult != 1.0:
        G.rendering_kwargs['depth_resolution'] = int(
            G.rendering_kwargs.get('depth_resolution', 48) * args.sample_mult
        )
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs.get('depth_resolution_importance', 48) * args.sample_mult
        )
    
    # Reload modules for modifications to take effect
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new
    
    print(f"\nModel configuration:")
    print(f"  Neural rendering resolution: {G.neural_rendering_resolution}")
    print(f"  Depth resolution: {G.rendering_kwargs.get('depth_resolution', 48)} (base: 48, multiplier: {args.sample_mult})")
    print(f"  Depth resolution importance: {G.rendering_kwargs.get('depth_resolution_importance', 48)} (base: 48, multiplier: {args.sample_mult})")
    print(f"  Box warp: {G.rendering_kwargs.get('box_warp', 'N/A')}")
    print(f"  White background: {G.rendering_kwargs.get('white_back', False)}")
    
    print(f"\nGeneration settings:")
    print(f"  Truncation psi: {args.truncation_psi}")
    print(f"  Truncation cutoff: 14")
    print(f"  Number of frames: {args.num_frames}")
    print(f"  Voxel resolution: {args.voxel_res}")
    print(f"  Save depth maps: {not args.no_depth}")
    print(f"  Generate heatmap: {not args.no_heatmap}")
    print(f"  Colormap: {args.colormap}")
    
    # Generate videos for each seed
    for seed in seeds:
        # Create seed-based subfolder
        seed_outdir = os.path.join(args.outdir, f'seed{seed}')
        os.makedirs(seed_outdir, exist_ok=True)
        
        output_path = os.path.join(seed_outdir, f'consistent_identity_seed{seed}.mp4')
        print(f"\nGenerating video for seed {seed} in {seed_outdir}...")
        
        try:
            generate_consistent_identity_video(
                G, output_path, evaluator, seed=seed, num_frames=args.num_frames,
                cfg=args.cfg, device=device, white_back=not args.no_white_back, nrr=args.nrr,
                save_depth=not args.no_depth, generate_heatmap=not args.no_heatmap,
                voxel_res=args.voxel_res, truncation_psi=args.truncation_psi,
                sample_mult=args.sample_mult, colormap=args.colormap
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\n" + "="*60)
                print("GPU OUT OF MEMORY ERROR")
                print("="*60)
                print("Even with minimal settings, the GPU ran out of memory.")
                print("\nSuggestions:")
                print("1. Reduce --num_frames (current: {})".format(args.num_frames))
                print("2. Reduce --nrr (current: {})".format(args.nrr))
                print("3. Reduce --voxel_res (current: {})".format(args.voxel_res))
                print("4. Reduce --sample_mult (current: {})".format(args.sample_mult))
                print("5. Try running with CPU: --device cpu")
                print("6. Close other GPU-using applications")
            else:
                raise
    
    print("\n" + "="*60)
    print("VIDEO GENERATION COMPLETE!")
    print("="*60)
    print(f"Generated {len(seeds)} video(s) with consistent identity")
    if not args.no_depth:
        print(f"Also saved depth maps for each video")
    print(f"Results saved to: {args.outdir}")
    
    # List generated files for first seed
    if len(seeds) > 0:
        first_seed = seeds[0]
        print(f"\nGenerated files for seed {first_seed} in seed{first_seed}/:")
        print(f"  - RGB video: consistent_identity_seed{first_seed}.mp4")
        print(f"  - RGB grid: consistent_identity_seed{first_seed}_grid.png")
        if not args.no_depth:
            print(f"  - Depth video: consistent_identity_seed{first_seed}_depth.mp4")
            print(f"  - Depth grid: consistent_identity_seed{first_seed}_depth_grid.png")
        if not args.no_heatmap:
            print(f"  - Heatmap mesh: consistent_identity_seed{first_seed}_heatmap_mesh.ply")
            print(f"  - Heatmap mesh video: consistent_identity_seed{first_seed}_heatmap_mesh_video.mp4")
            print(f"  - Heatmap mesh video grid: consistent_identity_seed{first_seed}_heatmap_mesh_video_grid.png")
            print(f"  - Score distribution: consistent_identity_seed{first_seed}_score_distribution.png")
            print(f"  - Comparison video: consistent_identity_seed{first_seed}_comparison.mp4")
    
    # Compare with the original inconsistent version if it exists
    original_path = os.path.join(args.outdir, 'rotating_heatmap_seed0.mp4')
    if os.path.exists(original_path) and 0 in seeds:
        print("\nComparison with original:")
        print(f"  Original (inconsistent): {original_path}")
        print(f"  New (consistent): {os.path.join(args.outdir, 'seed0', 'consistent_identity_seed0.mp4')}")
        print("\nThe new video should show the same face/identity throughout the rotation,")
        print("while the original shows different identities as the camera rotates.")


if __name__ == '__main__':
    # Note: This script should be run from the SphereHead root directory:
    # python synth_eval/evaluate_synthetic.py
    main()