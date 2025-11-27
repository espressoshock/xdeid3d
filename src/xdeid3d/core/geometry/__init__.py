"""
Spherical geometry utilities.

This module provides functions for working with spherical coordinates
on S², computing great-circle distances, and converting between
Cartesian and spherical coordinate systems.
"""

from xdeid3d.core.geometry.spherical import (
    great_circle_distance,
    angular_distance,
    angular_distance_degrees,
    cartesian_to_spherical,
    spherical_to_cartesian,
    normalize_angle,
    mesh_vertex_to_viewing_angle,
    SphericalPoint,
)

__all__ = [
    "great_circle_distance",
    "angular_distance",
    "angular_distance_degrees",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "normalize_angle",
    "mesh_vertex_to_viewing_angle",
    "SphericalPoint",
]