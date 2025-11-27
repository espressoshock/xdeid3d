"""
Camera pose sampling utilities for 3D synthesis.

Provides camera pose samplers and transformation utilities for
generating camera positions and orientations for 3D rendering.

The coordinate system uses:
- Y-axis: up
- Z-axis: forward (camera looks towards negative Z in camera space)
- X-axis: right

Angles are specified in radians:
- yaw (horizontal): rotation around Y axis, 0 = looking along Z axis
- pitch (vertical): angle from Y axis, π/2 = horizontal
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "CameraPoseSampler",
    "GaussianCameraSampler",
    "UniformCameraSampler",
    "LookAtCameraSampler",
    "create_look_at_matrix",
    "fov_to_intrinsics",
    "create_camera_matrix",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
]


@dataclass
class CameraPose:
    """Camera pose representation.

    Attributes:
        cam2world: 4x4 camera-to-world transformation matrix
        position: Camera position in world space
        yaw: Horizontal angle in radians
        pitch: Vertical angle in radians
        radius: Distance from origin
    """
    cam2world: np.ndarray
    position: np.ndarray
    yaw: float
    pitch: float
    radius: float

    @property
    def world2cam(self) -> np.ndarray:
        """Get world-to-camera transformation matrix."""
        return np.linalg.inv(self.cam2world)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "radius": self.radius,
            "position": self.position.tolist(),
        }


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector(s) to unit length."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-10)


def spherical_to_cartesian(
    yaw: Union[float, np.ndarray],
    pitch: Union[float, np.ndarray],
    radius: float = 1.0,
) -> np.ndarray:
    """Convert spherical coordinates to Cartesian.

    Args:
        yaw: Horizontal angle (azimuth) in radians
        pitch: Vertical angle (polar) in radians, 0=up, π=down
        radius: Distance from origin

    Returns:
        Cartesian coordinates [x, y, z]
    """
    yaw = np.asarray(yaw)
    pitch = np.asarray(pitch)

    # Convert angles to Cartesian
    sin_pitch = np.sin(pitch)
    cos_pitch = np.cos(pitch)
    sin_yaw = np.sin(yaw)
    cos_yaw = np.cos(yaw)

    x = radius * sin_pitch * sin_yaw
    y = radius * cos_pitch
    z = radius * sin_pitch * cos_yaw

    if yaw.ndim == 0:
        return np.array([x, y, z])
    return np.stack([x, y, z], axis=-1)


def cartesian_to_spherical(
    position: np.ndarray,
) -> Tuple[float, float, float]:
    """Convert Cartesian coordinates to spherical.

    Args:
        position: Cartesian coordinates [x, y, z]

    Returns:
        Tuple of (yaw, pitch, radius)
    """
    x, y, z = position
    radius = np.linalg.norm(position)

    if radius < 1e-10:
        return 0.0, math.pi / 2, 0.0

    pitch = np.arccos(np.clip(y / radius, -1.0, 1.0))
    yaw = np.arctan2(x, z)

    return float(yaw), float(pitch), float(radius)


def create_look_at_matrix(
    camera_position: np.ndarray,
    target: np.ndarray = None,
    up: np.ndarray = None,
) -> np.ndarray:
    """Create a camera-to-world transformation matrix.

    Args:
        camera_position: Camera position in world space
        target: Point the camera looks at (default: origin)
        up: Up vector (default: [0, 1, 0])

    Returns:
        4x4 camera-to-world transformation matrix
    """
    if target is None:
        target = np.zeros(3)
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    camera_position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Forward vector (camera looks towards -Z in camera space)
    forward = normalize_vector(target - camera_position)

    # Right vector
    right = normalize_vector(np.cross(up, forward))

    # Recalculate up to ensure orthogonality
    up = normalize_vector(np.cross(forward, right))

    # Build rotation matrix (columns are right, up, -forward in camera space)
    rotation = np.eye(4)
    rotation[:3, 0] = right
    rotation[:3, 1] = up
    rotation[:3, 2] = -forward

    # Build translation matrix
    translation = np.eye(4)
    translation[:3, 3] = camera_position

    # Combined transformation
    cam2world = translation @ rotation

    return cam2world


def fov_to_intrinsics(
    fov_degrees: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """Create camera intrinsics matrix from field of view.

    Args:
        fov_degrees: Vertical field of view in degrees
        width: Image width in pixels (optional, for pixel units)
        height: Image height in pixels (optional, for pixel units)

    Returns:
        3x3 camera intrinsics matrix. If width/height not provided,
        returns normalized intrinsics (principal point at 0.5, 0.5).
    """
    fov_radians = fov_degrees * math.pi / 180.0
    focal_length = 1.0 / (2.0 * math.tan(fov_radians / 2.0))

    if width is not None and height is not None:
        fx = focal_length * height
        fy = focal_length * height
        cx = width / 2.0
        cy = height / 2.0
    else:
        fx = fy = focal_length
        cx = cy = 0.5

    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    return intrinsics


def create_camera_matrix(
    yaw: float,
    pitch: float,
    radius: float = 1.0,
    target: np.ndarray = None,
) -> np.ndarray:
    """Create camera-to-world matrix from spherical coordinates.

    Args:
        yaw: Horizontal angle in radians
        pitch: Vertical angle in radians
        radius: Distance from target
        target: Point to look at (default: origin)

    Returns:
        4x4 camera-to-world transformation matrix
    """
    position = spherical_to_cartesian(yaw, pitch, radius)

    if target is not None:
        position = position + np.asarray(target)

    return create_look_at_matrix(position, target)


class CameraPoseSampler(ABC):
    """Abstract base class for camera pose samplers."""

    @abstractmethod
    def sample(self, batch_size: int = 1) -> Union[CameraPose, list]:
        """Sample camera pose(s).

        Args:
            batch_size: Number of poses to sample

        Returns:
            Single CameraPose if batch_size=1, else list of CameraPose
        """
        pass

    def __call__(self, batch_size: int = 1) -> Union[CameraPose, list]:
        """Sample camera pose(s)."""
        return self.sample(batch_size)


class GaussianCameraSampler(CameraPoseSampler):
    """Sample camera poses from a Gaussian distribution.

    Camera positions are sampled on a sphere around a target point,
    with yaw and pitch drawn from Gaussian distributions.

    Args:
        yaw_mean: Mean horizontal angle in radians
        pitch_mean: Mean vertical angle in radians (π/2 = horizontal)
        yaw_std: Standard deviation of yaw in radians
        pitch_std: Standard deviation of pitch in radians
        radius: Distance from target
        target: Point to look at (default: origin)
    """

    def __init__(
        self,
        yaw_mean: float = 0.0,
        pitch_mean: float = math.pi / 2,
        yaw_std: float = 0.0,
        pitch_std: float = 0.0,
        radius: float = 1.0,
        target: np.ndarray = None,
    ):
        self.yaw_mean = yaw_mean
        self.pitch_mean = pitch_mean
        self.yaw_std = yaw_std
        self.pitch_std = pitch_std
        self.radius = radius
        self.target = target if target is not None else np.zeros(3)

    def sample(self, batch_size: int = 1) -> Union[CameraPose, list]:
        """Sample camera pose(s) from Gaussian distribution."""
        poses = []

        for _ in range(batch_size):
            # Sample angles
            yaw = np.random.normal(self.yaw_mean, self.yaw_std)
            pitch = np.random.normal(self.pitch_mean, self.pitch_std)

            # Clamp pitch to valid range
            pitch = np.clip(pitch, 1e-5, math.pi - 1e-5)

            # Create camera matrix
            position = spherical_to_cartesian(yaw, pitch, self.radius)
            position = position + self.target

            cam2world = create_look_at_matrix(position, self.target)

            poses.append(CameraPose(
                cam2world=cam2world,
                position=position,
                yaw=float(yaw),
                pitch=float(pitch),
                radius=self.radius,
            ))

        return poses[0] if batch_size == 1 else poses


class UniformCameraSampler(CameraPoseSampler):
    """Sample camera poses from a uniform distribution.

    Camera positions are sampled uniformly within specified ranges
    around mean yaw and pitch values.

    Args:
        yaw_mean: Mean horizontal angle in radians
        pitch_mean: Mean vertical angle in radians
        yaw_range: Half-range for yaw sampling (±yaw_range)
        pitch_range: Half-range for pitch sampling (±pitch_range)
        radius: Distance from target
        target: Point to look at (default: origin)
    """

    def __init__(
        self,
        yaw_mean: float = 0.0,
        pitch_mean: float = math.pi / 2,
        yaw_range: float = 0.0,
        pitch_range: float = 0.0,
        radius: float = 1.0,
        target: np.ndarray = None,
    ):
        self.yaw_mean = yaw_mean
        self.pitch_mean = pitch_mean
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.radius = radius
        self.target = target if target is not None else np.zeros(3)

    def sample(self, batch_size: int = 1) -> Union[CameraPose, list]:
        """Sample camera pose(s) from uniform distribution."""
        poses = []

        for _ in range(batch_size):
            # Sample angles uniformly
            yaw = self.yaw_mean + np.random.uniform(-self.yaw_range, self.yaw_range)
            pitch = self.pitch_mean + np.random.uniform(-self.pitch_range, self.pitch_range)

            # Clamp pitch to valid range
            pitch = np.clip(pitch, 1e-5, math.pi - 1e-5)

            # Create camera matrix
            position = spherical_to_cartesian(yaw, pitch, self.radius)
            position = position + self.target

            cam2world = create_look_at_matrix(position, self.target)

            poses.append(CameraPose(
                cam2world=cam2world,
                position=position,
                yaw=float(yaw),
                pitch=float(pitch),
                radius=self.radius,
            ))

        return poses[0] if batch_size == 1 else poses


class LookAtCameraSampler(CameraPoseSampler):
    """Sample camera poses looking at a specific target.

    Generates a sequence of camera poses orbiting around a target point.

    Args:
        target: Point to look at (default: origin)
        radius: Distance from target
        num_frames: Number of frames for full orbit
        yaw_range: Range of yaw angles (default: full 2π)
        pitch_range: Range of pitch angles
        pitch_center: Center pitch angle (default: π/2 = horizontal)
    """

    def __init__(
        self,
        target: np.ndarray = None,
        radius: float = 1.0,
        num_frames: int = 120,
        yaw_range: Tuple[float, float] = (-math.pi, math.pi),
        pitch_range: Tuple[float, float] = None,
        pitch_center: float = math.pi / 2,
    ):
        self.target = target if target is not None else np.zeros(3)
        self.radius = radius
        self.num_frames = num_frames
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range or (pitch_center, pitch_center)
        self.pitch_center = pitch_center

    def sample(self, batch_size: int = 1) -> Union[CameraPose, list]:
        """Sample sequential camera pose(s)."""
        poses = []

        for i in range(batch_size):
            # Calculate angle for this frame
            t = i / max(batch_size - 1, 1)

            yaw = self.yaw_range[0] + t * (self.yaw_range[1] - self.yaw_range[0])
            pitch = self.pitch_range[0] + t * (self.pitch_range[1] - self.pitch_range[0])

            # Clamp pitch
            pitch = np.clip(pitch, 1e-5, math.pi - 1e-5)

            # Create camera matrix
            position = spherical_to_cartesian(yaw, pitch, self.radius)
            position = position + self.target

            cam2world = create_look_at_matrix(position, self.target)

            poses.append(CameraPose(
                cam2world=cam2world,
                position=position,
                yaw=float(yaw),
                pitch=float(pitch),
                radius=self.radius,
            ))

        return poses[0] if batch_size == 1 else poses

    def orbit(self, num_frames: int = None) -> list:
        """Generate a complete orbit of camera poses.

        Args:
            num_frames: Number of frames (default: self.num_frames)

        Returns:
            List of CameraPose for the orbit
        """
        if num_frames is None:
            num_frames = self.num_frames

        return self.sample(num_frames)
