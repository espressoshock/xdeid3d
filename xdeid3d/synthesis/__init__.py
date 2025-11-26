"""
Synthesis module for X-DeID3D.

Provides camera pose sampling, mesh extraction, and rendering utilities
for 3D face synthesis and visualization.
"""

from xdeid3d.synthesis.camera import (
    CameraPoseSampler,
    GaussianCameraSampler,
    UniformCameraSampler,
    create_look_at_matrix,
    fov_to_intrinsics,
    create_camera_matrix,
)
from xdeid3d.synthesis.mesh_extraction import (
    extract_mesh_from_volume,
    MarchingCubesExtractor,
    convert_sdf_to_mesh,
)
from xdeid3d.synthesis.rendering import (
    BasicRenderer,
    RenderConfig,
    render_mesh_to_image,
)

__all__ = [
    # Camera utilities
    "CameraPoseSampler",
    "GaussianCameraSampler",
    "UniformCameraSampler",
    "create_look_at_matrix",
    "fov_to_intrinsics",
    "create_camera_matrix",
    # Mesh extraction
    "extract_mesh_from_volume",
    "MarchingCubesExtractor",
    "convert_sdf_to_mesh",
    # Rendering
    "BasicRenderer",
    "RenderConfig",
    "render_mesh_to_image",
]
