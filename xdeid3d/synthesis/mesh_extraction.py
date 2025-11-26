"""
Mesh extraction utilities for 3D synthesis.

Provides functions and classes for extracting 3D meshes from
volumetric data using marching cubes and related algorithms.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "extract_mesh_from_volume",
    "MarchingCubesExtractor",
    "convert_sdf_to_mesh",
    "ExtractedMesh",
]


@dataclass
class ExtractedMesh:
    """Container for extracted mesh data.

    Attributes:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        normals: Nx3 array of vertex normals (optional)
        values: N array of vertex values (optional)
    """
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return len(self.faces)

    def transform(
        self,
        scale: float = 1.0,
        offset: np.ndarray = None,
        rotation: np.ndarray = None,
    ) -> "ExtractedMesh":
        """Apply transformation to mesh.

        Args:
            scale: Uniform scale factor
            offset: Translation offset [x, y, z]
            rotation: 3x3 rotation matrix

        Returns:
            New ExtractedMesh with transformation applied
        """
        vertices = self.vertices.copy()

        if rotation is not None:
            vertices = vertices @ rotation.T

        vertices = vertices * scale

        if offset is not None:
            vertices = vertices + np.asarray(offset)

        normals = self.normals
        if normals is not None and rotation is not None:
            normals = normals @ rotation.T

        return ExtractedMesh(
            vertices=vertices,
            faces=self.faces.copy(),
            normals=normals,
            values=self.values.copy() if self.values is not None else None,
        )

    def save_ply(self, path: Union[str, Path]) -> None:
        """Save mesh to PLY file.

        Args:
            path: Output file path
        """
        path = Path(path)

        with open(path, "wb") as f:
            # Write header
            header = [
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {self.n_vertices}",
                "property float x",
                "property float y",
                "property float z",
            ]

            if self.normals is not None:
                header.extend([
                    "property float nx",
                    "property float ny",
                    "property float nz",
                ])

            header.extend([
                f"element face {self.n_faces}",
                "property list uchar int vertex_indices",
                "end_header",
            ])

            f.write("\n".join(header).encode("ascii"))
            f.write(b"\n")

            # Write vertices
            if self.normals is not None:
                vertex_data = np.hstack([
                    self.vertices.astype(np.float32),
                    self.normals.astype(np.float32),
                ])
            else:
                vertex_data = self.vertices.astype(np.float32)

            f.write(vertex_data.tobytes())

            # Write faces
            for face in self.faces:
                f.write(np.array([3], dtype=np.uint8).tobytes())
                f.write(face.astype(np.int32).tobytes())

    def save_obj(self, path: Union[str, Path]) -> None:
        """Save mesh to OBJ file.

        Args:
            path: Output file path
        """
        path = Path(path)

        with open(path, "w") as f:
            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write normals
            if self.normals is not None:
                for n in self.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # Write faces (OBJ uses 1-indexed)
            for face in self.faces:
                if self.normals is not None:
                    f.write(f"f {face[0]+1}//{face[0]+1} "
                            f"{face[1]+1}//{face[1]+1} "
                            f"{face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def extract_mesh_from_volume(
    volume: np.ndarray,
    level: float = 0.0,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ExtractedMesh:
    """Extract mesh from 3D volume using marching cubes.

    Args:
        volume: 3D numpy array (NxMxK)
        level: Isosurface level
        spacing: Voxel spacing in each dimension
        origin: Origin of the volume in world space

    Returns:
        ExtractedMesh with vertices, faces, and normals
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError(
            "scikit-image is required for mesh extraction. "
            "Install with: pip install scikit-image"
        )

    # Run marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        volume,
        level=level,
        spacing=spacing,
    )

    # Transform to world coordinates
    origin = np.asarray(origin)
    verts = verts + origin

    return ExtractedMesh(
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        normals=normals.astype(np.float32),
        values=values.astype(np.float32),
    )


def convert_sdf_to_mesh(
    sdf: np.ndarray,
    voxel_size: float = 1.0,
    origin: Tuple[float, float, float] = None,
    level: float = 0.0,
    scale: float = None,
    offset: np.ndarray = None,
) -> ExtractedMesh:
    """Convert SDF (Signed Distance Field) to mesh.

    Args:
        sdf: 3D array of signed distance values
        voxel_size: Size of each voxel
        origin: Origin of the SDF grid (default: centered)
        level: Isosurface level
        scale: Optional scale factor to apply
        offset: Optional offset to apply

    Returns:
        ExtractedMesh extracted from the SDF
    """
    # Default origin: center the mesh
    if origin is None:
        shape = np.array(sdf.shape)
        origin = -shape * voxel_size / 2

    spacing = (voxel_size, voxel_size, voxel_size)

    mesh = extract_mesh_from_volume(
        sdf,
        level=level,
        spacing=spacing,
        origin=origin,
    )

    # Apply optional transformations
    if scale is not None or offset is not None:
        mesh = mesh.transform(
            scale=scale if scale is not None else 1.0,
            offset=offset,
        )

    return mesh


class MarchingCubesExtractor:
    """Marching cubes mesh extractor.

    Provides configurable mesh extraction from volumetric data.

    Args:
        level: Default isosurface level
        voxel_size: Default voxel size
        smooth: Whether to smooth the extracted mesh
        smooth_iterations: Number of smoothing iterations

    Example:
        >>> extractor = MarchingCubesExtractor(level=0.5)
        >>> mesh = extractor.extract(volume_data)
        >>> mesh.save_ply("output.ply")
    """

    def __init__(
        self,
        level: float = 0.0,
        voxel_size: float = 1.0,
        smooth: bool = False,
        smooth_iterations: int = 5,
    ):
        self.level = level
        self.voxel_size = voxel_size
        self.smooth = smooth
        self.smooth_iterations = smooth_iterations

    def extract(
        self,
        volume: np.ndarray,
        level: float = None,
        origin: Tuple[float, float, float] = None,
    ) -> ExtractedMesh:
        """Extract mesh from volume.

        Args:
            volume: 3D numpy array
            level: Isosurface level (default: self.level)
            origin: Grid origin (default: centered)

        Returns:
            Extracted mesh
        """
        if level is None:
            level = self.level

        mesh = convert_sdf_to_mesh(
            volume,
            voxel_size=self.voxel_size,
            origin=origin,
            level=level,
        )

        if self.smooth:
            mesh = self._smooth_mesh(mesh)

        return mesh

    def _smooth_mesh(self, mesh: ExtractedMesh) -> ExtractedMesh:
        """Apply Laplacian smoothing to mesh.

        Args:
            mesh: Input mesh

        Returns:
            Smoothed mesh
        """
        vertices = mesh.vertices.copy()

        # Build adjacency
        adjacency = [set() for _ in range(len(vertices))]
        for face in mesh.faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        adjacency[face[i]].add(face[j])

        # Laplacian smoothing
        for _ in range(self.smooth_iterations):
            new_vertices = np.zeros_like(vertices)
            for i, adj in enumerate(adjacency):
                if adj:
                    neighbors = np.array(list(adj))
                    new_vertices[i] = np.mean(vertices[neighbors], axis=0)
                else:
                    new_vertices[i] = vertices[i]
            vertices = 0.5 * vertices + 0.5 * new_vertices

        # Recompute normals
        normals = self._compute_normals(vertices, mesh.faces)

        return ExtractedMesh(
            vertices=vertices,
            faces=mesh.faces,
            normals=normals,
            values=mesh.values,
        )

    def _compute_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """Compute vertex normals from mesh.

        Args:
            vertices: Vertex positions
            faces: Face indices

        Returns:
            Vertex normals
        """
        normals = np.zeros_like(vertices)

        # Compute face normals and accumulate to vertices
        for face in faces:
            v0, v1, v2 = vertices[face]
            e1 = v1 - v0
            e2 = v2 - v0
            face_normal = np.cross(e1, e2)

            for idx in face:
                normals[idx] += face_normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-10)

        return normals

    def from_mrc(self, path: Union[str, Path], level: float = None) -> ExtractedMesh:
        """Extract mesh from MRC file.

        Args:
            path: Path to MRC file
            level: Isosurface level

        Returns:
            Extracted mesh
        """
        try:
            import mrcfile
        except ImportError:
            raise ImportError(
                "mrcfile is required to read MRC files. "
                "Install with: pip install mrcfile"
            )

        with mrcfile.open(str(path)) as mrc:
            # MRC files store data in (z, y, x) order
            volume = np.transpose(mrc.data, (2, 1, 0))

        return self.extract(volume, level=level)
