"""
Basic rendering utilities for 3D synthesis.

Provides simple rendering functionality for visualizing 3D meshes
without requiring a full neural rendering pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "BasicRenderer",
    "RenderConfig",
    "render_mesh_to_image",
    "RenderResult",
]


@dataclass
class RenderConfig:
    """Configuration for rendering.

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        fov: Field of view in degrees
        near: Near clipping plane
        far: Far clipping plane
        background: Background color RGB (0-255)
        ambient: Ambient light intensity
        diffuse: Diffuse light intensity
        specular: Specular light intensity
        light_direction: Direction of light source
    """
    width: int = 512
    height: int = 512
    fov: float = 45.0
    near: float = 0.01
    far: float = 100.0
    background: Tuple[int, int, int] = (255, 255, 255)
    ambient: float = 0.3
    diffuse: float = 0.7
    specular: float = 0.2
    light_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class RenderResult:
    """Result of a render operation.

    Attributes:
        image: RGB image array (H, W, 3)
        depth: Depth map (H, W) if available
        mask: Binary mask (H, W) if available
    """
    image: np.ndarray
    depth: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    def save(self, path: Union[str, Path], include_depth: bool = False) -> None:
        """Save render result to file.

        Args:
            path: Output path (PNG, JPG, etc.)
            include_depth: Save depth map alongside
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for saving images")

        path = Path(path)

        # Save main image
        img = Image.fromarray(self.image.astype(np.uint8))
        img.save(path)

        # Save depth if requested
        if include_depth and self.depth is not None:
            depth_path = path.with_stem(path.stem + "_depth")
            # Normalize depth for visualization
            depth_norm = self.depth.copy()
            valid = np.isfinite(depth_norm)
            if valid.any():
                depth_min = depth_norm[valid].min()
                depth_max = depth_norm[valid].max()
                depth_norm = (depth_norm - depth_min) / (depth_max - depth_min + 1e-8)
                depth_norm = (depth_norm * 255).astype(np.uint8)
                depth_img = Image.fromarray(depth_norm)
                depth_img.save(depth_path)


class BasicRenderer:
    """Basic software renderer for 3D meshes.

    Provides simple rasterization-based rendering for visualization.
    For high-quality neural rendering, use dedicated rendering backends.

    Args:
        config: Render configuration

    Example:
        >>> renderer = BasicRenderer(RenderConfig(width=512, height=512))
        >>> result = renderer.render(mesh, camera_matrix)
        >>> result.save("output.png")
    """

    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
        self._trimesh_available = None

    def _check_trimesh(self) -> bool:
        """Check if trimesh is available."""
        if self._trimesh_available is None:
            try:
                import trimesh
                self._trimesh_available = True
            except ImportError:
                self._trimesh_available = False
        return self._trimesh_available

    def render(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        cam2world: np.ndarray,
        vertex_colors: np.ndarray = None,
    ) -> RenderResult:
        """Render mesh to image.

        Args:
            vertices: Nx3 vertex positions
            faces: Mx3 face indices
            cam2world: 4x4 camera-to-world transformation
            vertex_colors: Nx3 vertex colors (0-255) or None

        Returns:
            RenderResult with rendered image
        """
        if self._check_trimesh():
            return self._render_trimesh(vertices, faces, cam2world, vertex_colors)
        else:
            return self._render_simple(vertices, faces, cam2world, vertex_colors)

    def _render_trimesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        cam2world: np.ndarray,
        vertex_colors: np.ndarray = None,
    ) -> RenderResult:
        """Render using trimesh/pyrender."""
        try:
            import trimesh
            import trimesh.viewer
        except ImportError:
            return self._render_simple(vertices, faces, cam2world, vertex_colors)

        # Create mesh
        if vertex_colors is not None:
            colors = np.hstack([
                vertex_colors,
                np.full((len(vertex_colors), 1), 255, dtype=np.uint8)
            ])
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=colors,
            )
        else:
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
            )

        # Create scene
        scene = trimesh.Scene([mesh])

        # Set camera
        world2cam = np.linalg.inv(cam2world)

        # Try to render with pyrender if available
        try:
            import pyrender

            # Create pyrender scene
            pr_scene = pyrender.Scene(
                bg_color=np.array(self.config.background + (255,)) / 255.0,
                ambient_light=np.full(3, self.config.ambient),
            )

            # Add mesh
            if vertex_colors is not None:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh)
            else:
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                )
                pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            pr_scene.add(pr_mesh)

            # Add camera
            camera = pyrender.PerspectiveCamera(
                yfov=np.radians(self.config.fov),
                aspectRatio=self.config.width / self.config.height,
                znear=self.config.near,
                zfar=self.config.far,
            )
            pr_scene.add(camera, pose=cam2world)

            # Add light
            light_dir = np.array(self.config.light_direction)
            light = pyrender.DirectionalLight(
                color=np.ones(3),
                intensity=self.config.diffuse * 3,
            )
            light_pose = np.eye(4)
            light_pose[:3, 2] = -light_dir
            pr_scene.add(light, pose=light_pose)

            # Render
            renderer = pyrender.OffscreenRenderer(
                self.config.width,
                self.config.height,
            )
            color, depth = renderer.render(pr_scene)
            renderer.delete()

            mask = depth > 0

            return RenderResult(
                image=color,
                depth=depth,
                mask=mask,
            )

        except ImportError:
            # Fall back to simple rendering
            return self._render_simple(vertices, faces, cam2world, vertex_colors)

    def _render_simple(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        cam2world: np.ndarray,
        vertex_colors: np.ndarray = None,
    ) -> RenderResult:
        """Simple software rendering fallback.

        This provides basic wireframe/point rendering when
        proper rendering libraries are not available.
        """
        width = self.config.width
        height = self.config.height

        # Initialize image with background
        image = np.full((height, width, 3), self.config.background, dtype=np.uint8)
        depth = np.full((height, width), np.inf, dtype=np.float32)

        # Camera matrices
        world2cam = np.linalg.inv(cam2world)

        # Intrinsics
        fov_rad = np.radians(self.config.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        aspect = width / height

        # Project vertices
        verts_h = np.hstack([vertices, np.ones((len(vertices), 1))])
        verts_cam = (world2cam @ verts_h.T).T[:, :3]

        # Perspective projection
        x = verts_cam[:, 0] / (-verts_cam[:, 2] + 1e-8) * f / aspect
        y = verts_cam[:, 1] / (-verts_cam[:, 2] + 1e-8) * f
        z = -verts_cam[:, 2]

        # Convert to pixel coordinates
        px = ((x + 1) * 0.5 * width).astype(int)
        py = ((1 - y) * 0.5 * height).astype(int)

        # Simple point rendering
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height) & (z > 0)

        if vertex_colors is not None:
            colors = vertex_colors[valid]
        else:
            # Default gray
            colors = np.full((valid.sum(), 3), 180, dtype=np.uint8)

        for i, (p_x, p_y, p_z) in enumerate(zip(px[valid], py[valid], z[valid])):
            if p_z < depth[p_y, p_x]:
                depth[p_y, p_x] = p_z
                image[p_y, p_x] = colors[i]

        mask = np.isfinite(depth) & (depth < np.inf)

        return RenderResult(
            image=image,
            depth=np.where(mask, depth, 0),
            mask=mask,
        )

    def render_orbit(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        num_frames: int = 120,
        radius: float = 2.5,
        pitch: float = np.pi / 2,
        vertex_colors: np.ndarray = None,
    ) -> List[RenderResult]:
        """Render mesh from orbiting camera positions.

        Args:
            vertices: Nx3 vertex positions
            faces: Mx3 face indices
            num_frames: Number of frames in orbit
            radius: Camera distance from origin
            pitch: Camera pitch angle (π/2 = horizontal)
            vertex_colors: Optional vertex colors

        Returns:
            List of RenderResults
        """
        from xdeid3d.synthesis.camera import (
            LookAtCameraSampler,
        )

        sampler = LookAtCameraSampler(
            radius=radius,
            num_frames=num_frames,
            pitch_range=(pitch, pitch),
        )

        poses = sampler.orbit(num_frames)
        results = []

        for pose in poses:
            result = self.render(
                vertices,
                faces,
                pose.cam2world,
                vertex_colors,
            )
            results.append(result)

        return results


def render_mesh_to_image(
    vertices: np.ndarray,
    faces: np.ndarray,
    cam2world: np.ndarray,
    width: int = 512,
    height: int = 512,
    vertex_colors: np.ndarray = None,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Convenience function to render mesh to image.

    Args:
        vertices: Nx3 vertex positions
        faces: Mx3 face indices
        cam2world: 4x4 camera-to-world matrix
        width: Image width
        height: Image height
        vertex_colors: Optional Nx3 vertex colors
        background: Background color RGB

    Returns:
        RGB image array (H, W, 3)
    """
    config = RenderConfig(
        width=width,
        height=height,
        background=background,
    )

    renderer = BasicRenderer(config)
    result = renderer.render(vertices, faces, cam2world, vertex_colors)

    return result.image
