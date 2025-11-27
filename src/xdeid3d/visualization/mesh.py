"""
3D mesh export with metric-based vertex coloring.

This module provides tools for creating colored PLY meshes
from evaluation scores mapped to viewing angles.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from xdeid3d.visualization.colormaps import ColorMapper, apply_colormap
from xdeid3d.visualization.heatmaps import HeatmapGenerator, SphericalHeatmap

__all__ = [
    "MeshExporter",
    "ColoredMesh",
    "write_ply",
    "vertex_scores_from_angles",
]


@dataclass
class ColoredMesh:
    """
    Container for colored 3D mesh data.

    Attributes:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        vertex_colors: Nx3 array of RGB colors (0-255)
        vertex_scores: N array of raw scores
        normals: Optional Nx3 array of vertex normals
        metadata: Additional metadata

    Example:
        >>> mesh = ColoredMesh(
        ...     vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        ...     faces=np.array([[0, 1, 2]]),
        ...     vertex_colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
        ...     vertex_scores=np.array([0.2, 0.5, 0.8]),
        ... )
    """

    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray
    vertex_scores: np.ndarray
    normals: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return len(self.faces)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_corner, max_corner) bounding box."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def center(self) -> np.ndarray:
        """Return mesh center."""
        return self.vertices.mean(axis=0)

    def compute_normals(self) -> np.ndarray:
        """Compute vertex normals from face geometry."""
        normals = np.zeros_like(self.vertices)

        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)

            # Add to all vertices of this face
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.normals = normals / norms

        return self.normals

    def save_ply(self, path: Union[str, Path]) -> None:
        """Save to PLY file."""
        write_ply(
            path,
            vertices=self.vertices,
            faces=self.faces,
            vertex_colors=self.vertex_colors,
            normals=self.normals,
        )

    @classmethod
    def from_ply(cls, path: Union[str, Path]) -> "ColoredMesh":
        """Load from PLY file."""
        vertices, faces, colors, normals = read_ply(path)
        return cls(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            vertex_scores=np.mean(colors, axis=1) / 255.0,  # Approximate
            normals=normals,
        )


def write_ply(
    path: Union[str, Path],
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    binary: bool = True,
) -> None:
    """
    Write PLY file with optional vertex colors and normals.

    Args:
        path: Output file path
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        vertex_colors: Optional Nx3 array of RGB colors (0-255)
        normals: Optional Nx3 array of vertex normals
        binary: Use binary format (smaller, faster)

    Example:
        >>> write_ply(
        ...     "mesh.ply",
        ...     vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        ...     faces=np.array([[0, 1, 2]]),
        ...     vertex_colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
        ... )
    """
    path = Path(path)
    n_vertices = len(vertices)
    n_faces = len(faces)

    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0" if binary else "format ascii 1.0",
        f"element vertex {n_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if normals is not None:
        header_lines.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])

    if vertex_colors is not None:
        header_lines.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header_lines.extend([
        f"element face {n_faces}",
        "property list uchar int vertex_indices",
        "end_header",
    ])

    header = "\n".join(header_lines) + "\n"

    if binary:
        _write_ply_binary(
            path, header, vertices, faces, vertex_colors, normals
        )
    else:
        _write_ply_ascii(
            path, header, vertices, faces, vertex_colors, normals
        )


def _write_ply_binary(
    path: Path,
    header: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray],
    normals: Optional[np.ndarray],
) -> None:
    """Write binary PLY."""
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))

        # Write vertices
        for i in range(len(vertices)):
            # Position
            f.write(np.array(vertices[i], dtype='<f4').tobytes())

            # Normals
            if normals is not None:
                f.write(np.array(normals[i], dtype='<f4').tobytes())

            # Colors
            if vertex_colors is not None:
                f.write(np.array(vertex_colors[i], dtype='<u1').tobytes())

        # Write faces
        for face in faces:
            f.write(np.array([3], dtype='<u1').tobytes())  # 3 vertices per face
            f.write(np.array(face, dtype='<i4').tobytes())


def _write_ply_ascii(
    path: Path,
    header: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray],
    normals: Optional[np.ndarray],
) -> None:
    """Write ASCII PLY."""
    with open(path, 'w') as f:
        f.write(header)

        # Write vertices
        for i in range(len(vertices)):
            line_parts = [f"{vertices[i, 0]:.6f}", f"{vertices[i, 1]:.6f}", f"{vertices[i, 2]:.6f}"]

            if normals is not None:
                line_parts.extend([
                    f"{normals[i, 0]:.6f}",
                    f"{normals[i, 1]:.6f}",
                    f"{normals[i, 2]:.6f}",
                ])

            if vertex_colors is not None:
                line_parts.extend([
                    str(int(vertex_colors[i, 0])),
                    str(int(vertex_colors[i, 1])),
                    str(int(vertex_colors[i, 2])),
                ])

            f.write(" ".join(line_parts) + "\n")

        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def read_ply(
    path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read PLY file.

    Args:
        path: Input file path

    Returns:
        Tuple of (vertices, faces, vertex_colors, normals)
    """
    path = Path(path)

    # Try using plyfile if available
    try:
        from plyfile import PlyData

        plydata = PlyData.read(str(path))
        vertex_data = plydata['vertex']

        # Extract vertices
        vertices = np.column_stack([
            vertex_data['x'],
            vertex_data['y'],
            vertex_data['z'],
        ])

        # Extract colors if present
        colors = None
        if 'red' in vertex_data.data.dtype.names:
            colors = np.column_stack([
                vertex_data['red'],
                vertex_data['green'],
                vertex_data['blue'],
            ]).astype(np.uint8)

        # Extract normals if present
        normals = None
        if 'nx' in vertex_data.data.dtype.names:
            normals = np.column_stack([
                vertex_data['nx'],
                vertex_data['ny'],
                vertex_data['nz'],
            ])

        # Extract faces
        face_data = plydata['face']
        faces = np.vstack(face_data['vertex_indices'])

        return vertices, faces, colors, normals

    except ImportError:
        # Fallback to simple parser
        return _read_ply_simple(path)


def _read_ply_simple(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Simple PLY reader (ASCII only)."""
    with open(path, 'r') as f:
        lines = f.readlines()

    # Parse header
    n_vertices = 0
    n_faces = 0
    has_colors = False
    has_normals = False
    header_end = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('element vertex'):
            n_vertices = int(line.split()[-1])
        elif line.startswith('element face'):
            n_faces = int(line.split()[-1])
        elif 'red' in line:
            has_colors = True
        elif 'nx' in line:
            has_normals = True
        elif line == 'end_header':
            header_end = i + 1
            break

    # Parse data
    vertices = []
    colors = [] if has_colors else None
    normals = [] if has_normals else None
    faces = []

    vertex_lines = lines[header_end:header_end + n_vertices]
    face_lines = lines[header_end + n_vertices:header_end + n_vertices + n_faces]

    for line in vertex_lines:
        parts = line.strip().split()
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

        idx = 3
        if has_normals:
            normals.append([float(parts[idx]), float(parts[idx + 1]), float(parts[idx + 2])])
            idx += 3

        if has_colors:
            colors.append([int(parts[idx]), int(parts[idx + 1]), int(parts[idx + 2])])

    for line in face_lines:
        parts = line.strip().split()
        # Skip the vertex count (first element)
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    colors = np.array(colors, dtype=np.uint8) if colors else None
    normals = np.array(normals, dtype=np.float32) if normals else None

    return vertices, faces, colors, normals


def vertex_scores_from_angles(
    vertices: np.ndarray,
    angle_scores: Dict[Tuple[float, float], float],
    bandwidth: float = 0.5,
) -> np.ndarray:
    """
    Compute vertex scores from angle-based scores.

    For each vertex, compute its viewing direction and interpolate
    scores from nearby angle measurements.

    Args:
        vertices: Nx3 array of vertex positions
        angle_scores: Dict mapping (yaw, pitch) -> score
        bandwidth: Kernel bandwidth for interpolation

    Returns:
        N array of interpolated scores
    """
    n_vertices = len(vertices)
    scores = np.zeros(n_vertices)

    if not angle_scores:
        return np.full(n_vertices, 0.5)

    # Convert to arrays for vectorized computation
    angles = np.array(list(angle_scores.keys()))  # Mx2
    values = np.array(list(angle_scores.values()))  # M

    # Compute viewing direction for each vertex
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    directions = vertices / norms

    # Convert to spherical coordinates
    vertex_yaws = np.arctan2(directions[:, 2], directions[:, 0]) + np.pi  # [0, 2pi]
    vertex_pitches = np.arccos(np.clip(directions[:, 1], -1, 1))  # [0, pi]

    # Interpolate scores using kernel regression
    for i in range(n_vertices):
        # Angular distance on sphere
        yaw_diff = np.minimum(
            np.abs(angles[:, 0] - vertex_yaws[i]),
            2 * np.pi - np.abs(angles[:, 0] - vertex_yaws[i])
        )
        pitch_diff = np.abs(angles[:, 1] - vertex_pitches[i])

        # Combined angular distance (simplified)
        angular_dist = np.sqrt(yaw_diff ** 2 + pitch_diff ** 2)

        # Gaussian weighting
        weights = np.exp(-angular_dist ** 2 / (2 * bandwidth ** 2))
        total_weight = np.sum(weights)

        if total_weight > 1e-10:
            scores[i] = np.sum(values * weights) / total_weight
        else:
            scores[i] = np.mean(values)

    return scores


class MeshExporter:
    """
    Export 3D meshes colored by evaluation metrics.

    Handles conversion from evaluation results to colored PLY meshes
    suitable for 3D visualization.

    Args:
        colormap: Colormap for score-to-color conversion
        bandwidth: Kernel bandwidth for angle interpolation
        normalize_scores: Whether to normalize scores to [0, 1]

    Example:
        >>> exporter = MeshExporter(colormap="magma")
        >>> exporter.add_score(yaw=0, pitch=np.pi/2, score=0.8)
        >>> mesh = exporter.color_mesh(vertices, faces)
        >>> mesh.save_ply("output.ply")
    """

    def __init__(
        self,
        colormap: str = "magma",
        bandwidth: float = 0.5,
        normalize_scores: bool = True,
    ):
        self.colormap = colormap
        self.bandwidth = bandwidth
        self.normalize_scores = normalize_scores

        self._angle_scores: Dict[Tuple[float, float], float] = {}
        self._metric_name: str = "score"

    def add_score(
        self,
        yaw: float,
        pitch: float,
        score: float,
    ) -> None:
        """
        Add a score at a specific viewing angle.

        Args:
            yaw: Yaw angle in radians
            pitch: Pitch angle in radians
            score: Metric score value
        """
        if not np.isnan(score) and not np.isinf(score):
            # Quantize angles slightly for deduplication
            yaw_key = round(yaw, 4)
            pitch_key = round(pitch, 4)
            self._angle_scores[(yaw_key, pitch_key)] = float(score)

    def add_scores_from_evaluations(
        self,
        evaluations: List[Dict[str, Any]],
        metric_key: str = "score",
    ) -> None:
        """
        Add scores from evaluation result dictionaries.

        Args:
            evaluations: List of dicts with 'yaw', 'pitch', and metric_key
            metric_key: Key for score value in dict
        """
        for eval_data in evaluations:
            if 'yaw' in eval_data and 'pitch' in eval_data:
                score = eval_data.get(metric_key, 0)
                self.add_score(eval_data['yaw'], eval_data['pitch'], score)

    def set_metric_name(self, name: str) -> None:
        """Set the metric name for metadata."""
        self._metric_name = name

    @property
    def n_samples(self) -> int:
        """Number of angle-score samples."""
        return len(self._angle_scores)

    def clear(self) -> None:
        """Clear all stored scores."""
        self._angle_scores.clear()

    def color_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> ColoredMesh:
        """
        Color mesh vertices based on viewing angle scores.

        Args:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices
            vmin: Minimum score for normalization
            vmax: Maximum score for normalization

        Returns:
            ColoredMesh with vertex colors
        """
        # Compute vertex scores
        vertex_scores = vertex_scores_from_angles(
            vertices,
            self._angle_scores,
            bandwidth=self.bandwidth,
        )

        # Normalize if requested
        if self.normalize_scores:
            if vmin is None:
                vmin = vertex_scores.min()
            if vmax is None:
                vmax = vertex_scores.max()

            if vmax - vmin > 1e-10:
                vertex_scores = (vertex_scores - vmin) / (vmax - vmin)
            else:
                vertex_scores = np.full_like(vertex_scores, 0.5)

        # Apply colormap
        vertex_colors = apply_colormap(
            vertex_scores,
            colormap=self.colormap,
            as_uint8=True,
        )

        # Create mesh
        mesh = ColoredMesh(
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            vertex_colors=vertex_colors,
            vertex_scores=vertex_scores,
            metadata={
                'metric_name': self._metric_name,
                'colormap': self.colormap,
                'n_angle_samples': self.n_samples,
                'bandwidth': self.bandwidth,
                'vmin': vmin,
                'vmax': vmax,
            },
        )

        # Compute normals
        mesh.compute_normals()

        return mesh

    def export_ply(
        self,
        path: Union[str, Path],
        vertices: np.ndarray,
        faces: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> ColoredMesh:
        """
        Color mesh and export to PLY file.

        Args:
            path: Output file path
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices
            vmin: Minimum score for normalization
            vmax: Maximum score for normalization

        Returns:
            The colored mesh
        """
        mesh = self.color_mesh(vertices, faces, vmin, vmax)
        mesh.save_ply(path)
        return mesh

    def create_heatmap(self, resolution: int = 72) -> SphericalHeatmap:
        """
        Create a spherical heatmap from the stored scores.

        Args:
            resolution: Grid resolution

        Returns:
            SphericalHeatmap
        """
        generator = HeatmapGenerator(
            resolution=resolution,
            bandwidth=self.bandwidth,
        )
        generator.set_metric_name(self._metric_name)

        for (yaw, pitch), score in self._angle_scores.items():
            generator.add_score(yaw, pitch, score)

        return generator.generate()
