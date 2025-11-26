"""
Visualization utilities for X-DeID3D.

This package provides tools for generating visual representations
of anonymization evaluation results, including:

- Spherical heatmaps for pose-varying performance
- Colormap utilities for consistent visualization
- 2D and 3D projection methods

Sub-modules:
    - heatmaps: Spherical heatmap generation and interpolation
    - colormaps: Color mapping utilities
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
]
