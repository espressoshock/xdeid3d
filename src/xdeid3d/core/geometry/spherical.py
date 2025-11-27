"""
Spherical geometry utilities for kernel regression on S².

This module provides functions for working with spherical coordinates,
computing great-circle distances, and converting between coordinate systems.
"""

from typing import Tuple, Union
import numpy as np

__all__ = [
    "great_circle_distance",
    "angular_distance",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "normalize_angle",
    "SphericalPoint",
]


class SphericalPoint:
    """
    A point on the unit sphere S² in spherical coordinates.

    Attributes:
        theta: Azimuthal angle (yaw) in radians, range [0, 2π)
        phi: Polar angle (pitch/colatitude) in radians, range [0, π]

    Convention:
        - theta = 0 points along positive x-axis
        - phi = 0 is the north pole (positive y-axis)
        - phi = π/2 is the equator
        - phi = π is the south pole (negative y-axis)
    """

    __slots__ = ("theta", "phi")

    def __init__(self, theta: float, phi: float):
        self.theta = normalize_angle(theta, period=2 * np.pi)
        self.phi = np.clip(phi, 0.0, np.pi)

    @classmethod
    def from_cartesian(cls, x: float, y: float, z: float) -> "SphericalPoint":
        """Create spherical point from Cartesian coordinates."""
        theta, phi = cartesian_to_spherical(x, y, z)
        return cls(theta, phi)

    @classmethod
    def from_degrees(cls, theta_deg: float, phi_deg: float) -> "SphericalPoint":
        """Create spherical point from angles in degrees."""
        return cls(np.radians(theta_deg), np.radians(phi_deg))

    def to_cartesian(self) -> Tuple[float, float, float]:
        """Convert to Cartesian coordinates on unit sphere."""
        return spherical_to_cartesian(self.theta, self.phi)

    def to_degrees(self) -> Tuple[float, float]:
        """Return angles in degrees."""
        return np.degrees(self.theta), np.degrees(self.phi)

    def distance_to(self, other: "SphericalPoint") -> float:
        """Compute great-circle distance to another point."""
        return great_circle_distance(self.theta, self.phi, other.theta, other.phi)

    def __repr__(self) -> str:
        return f"SphericalPoint(θ={np.degrees(self.theta):.1f}°, φ={np.degrees(self.phi):.1f}°)"


def normalize_angle(angle: Union[float, np.ndarray], period: float = 2 * np.pi) -> Union[float, np.ndarray]:
    """
    Normalize angle(s) to range [0, period).

    Args:
        angle: Angle(s) in radians
        period: Period for normalization (default 2π)

    Returns:
        Normalized angle(s)
    """
    return angle % period


def cartesian_to_spherical(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert Cartesian coordinates to spherical (theta, phi).

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        theta: Azimuthal angle in [0, 2π)
        phi: Polar angle (colatitude) in [0, π]

    Convention:
        - theta measured from positive x-axis in xz-plane
        - phi measured from positive y-axis
    """
    # Compute radius for normalization
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-8)  # Avoid division by zero

    # Azimuthal angle (yaw)
    theta = np.arctan2(z, x) + np.pi  # Shift to [0, 2π)

    # Polar angle (colatitude from y-axis)
    phi = np.arccos(np.clip(y / r, -1.0, 1.0))

    return theta, phi


def spherical_to_cartesian(
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    r: Union[float, np.ndarray] = 1.0,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert spherical coordinates to Cartesian.

    Args:
        theta: Azimuthal angle (yaw) in radians
        phi: Polar angle (colatitude) in radians
        r: Radius (default 1.0 for unit sphere)

    Returns:
        x, y, z: Cartesian coordinates
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.cos(phi)
    z = r * np.sin(phi) * np.sin(theta)
    return x, y, z


def great_circle_distance(
    theta1: Union[float, np.ndarray],
    phi1: Union[float, np.ndarray],
    theta2: Union[float, np.ndarray],
    phi2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute great-circle distance between points on unit sphere.

    Uses the Vincenty formula for numerical stability:
        d = atan2(
            sqrt((cos(φ₂)sin(Δλ))² + (cos(φ₁)sin(φ₂) - sin(φ₁)cos(φ₂)cos(Δλ))²),
            sin(φ₁)sin(φ₂) + cos(φ₁)cos(φ₂)cos(Δλ)
        )

    Note: We use colatitude convention where phi is measured from pole.

    Args:
        theta1, phi1: First point (azimuth, colatitude) in radians
        theta2, phi2: Second point (azimuth, colatitude) in radians

    Returns:
        Angular distance in radians, range [0, π]
    """
    # Azimuthal difference
    delta_theta = theta2 - theta1

    # Use spherical law of cosines with colatitude
    # cos(d) = sin(φ₁)sin(φ₂)cos(Δθ) + cos(φ₁)cos(φ₂)
    # where φ is colatitude (measured from pole)
    cos_dist = (
        np.sin(phi1) * np.sin(phi2) * np.cos(delta_theta) +
        np.cos(phi1) * np.cos(phi2)
    )

    # Clamp to valid range to handle numerical errors
    cos_dist = np.clip(cos_dist, -1.0, 1.0)

    return np.arccos(cos_dist)


def angular_distance(
    yaw1: Union[float, np.ndarray],
    pitch1: Union[float, np.ndarray],
    yaw2: Union[float, np.ndarray],
    pitch2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute angular distance using camera pose convention.

    This is an alias for great_circle_distance with more intuitive naming
    for camera/view angle use cases.

    Args:
        yaw1, pitch1: First viewing angle (yaw, pitch) in radians
        yaw2, pitch2: Second viewing angle (yaw, pitch) in radians

    Returns:
        Angular distance in radians
    """
    return great_circle_distance(yaw1, pitch1, yaw2, pitch2)


def angular_distance_degrees(
    yaw1_deg: Union[float, np.ndarray],
    pitch1_deg: Union[float, np.ndarray],
    yaw2_deg: Union[float, np.ndarray],
    pitch2_deg: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute angular distance from angles in degrees.

    Convenience function that handles degree-to-radian conversion
    and accounts for periodic yaw wrapping.

    Args:
        yaw1_deg, pitch1_deg: First viewing angle in degrees
        yaw2_deg, pitch2_deg: Second viewing angle in degrees

    Returns:
        Angular distance in radians
    """
    return great_circle_distance(
        np.radians(yaw1_deg % 360),
        np.radians(pitch1_deg),
        np.radians(yaw2_deg % 360),
        np.radians(pitch2_deg),
    )


def mesh_vertex_to_viewing_angle(
    vertex: np.ndarray,
    origin: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Convert mesh vertex position to viewing angle.

    Computes the spherical coordinates of the direction from origin
    to the vertex, which corresponds to the camera angle that would
    best view that vertex.

    Args:
        vertex: 3D vertex position (x, y, z)
        origin: Origin point (default: world origin [0, 0, 0])

    Returns:
        yaw, pitch: Viewing angle in radians
    """
    if origin is None:
        origin = np.zeros(3)

    direction = vertex - origin
    norm = np.linalg.norm(direction)

    if norm < 1e-8:
        return 0.0, np.pi / 2  # Default: front view

    direction = direction / norm
    return cartesian_to_spherical(direction[0], direction[1], direction[2])
