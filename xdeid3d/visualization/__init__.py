"""
Visualization utilities for X-DeID3D.

This package provides tools for generating visual representations
of anonymization evaluation results, including:

- Spherical heatmaps for pose-varying performance
- Colormap utilities for consistent visualization
- 2D and 3D projection methods
- 3D mesh export with vertex coloring

Sub-modules:
    - heatmaps: Spherical heatmap generation and interpolation
    - colormaps: Color mapping utilities
    - mesh: 3D mesh export with vertex colors
"""

from xdeid3d.visualization.colormaps import (
    ColorMapper,
    apply_colormap,
    get_colormap,
    normalize_scores,
    COLORMAPS,
)
from xdeid3d.visualization.heatmaps import (
    SphericalHeatmap,
    HeatmapGenerator,
    create_2d_heatmap,
    create_polar_heatmap,
    create_mollweide_projection,
)
from xdeid3d.visualization.mesh import (
    ColoredMesh,
    MeshExporter,
    write_ply,
    read_ply,
    vertex_scores_from_angles,
)

__all__ = [
    # Colormaps
    "ColorMapper",
    "apply_colormap",
    "get_colormap",
    "normalize_scores",
    "COLORMAPS",
    # Heatmaps
    "SphericalHeatmap",
    "HeatmapGenerator",
    "create_2d_heatmap",
    "create_polar_heatmap",
    "create_mollweide_projection",
    # Mesh
    "ColoredMesh",
    "MeshExporter",
    "write_ply",
    "read_ply",
    "vertex_scores_from_angles",
]
