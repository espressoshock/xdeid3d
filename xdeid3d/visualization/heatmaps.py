"""
Spherical heatmap generation for 3D visualization.

This module provides tools for generating 2D and 3D heatmaps
from evaluation scores mapped to viewing angles on S².
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from xdeid3d.visualization.colormaps import ColorMapper, apply_colormap, normalize_scores

__all__ = [
    "SphericalHeatmap",
    "HeatmapGenerator",
    "create_2d_heatmap",
    "create_polar_heatmap",
    "create_mollweide_projection",
]


@dataclass
class SphericalHeatmap:
    """
    Container for spherical heatmap data.

    Stores gridded scores on S² with associated coordinates.

    Attributes:
        yaw_grid: 2D array of yaw angles (radians)
        pitch_grid: 2D array of pitch angles (radians)
        score_grid: 2D array of interpolated scores
        resolution: Grid resolution (number of cells)
        metric_name: Name of the metric
        metadata: Additional metadata

    Example:
        >>> heatmap = SphericalHeatmap(
        ...     yaw_grid=np.zeros((72, 72)),
        ...     pitch_grid=np.zeros((72, 72)),
        ...     score_grid=np.random.rand(72, 72),
        ...     resolution=72,
        ...     metric_name="arcface_cosine_distance"
        ... )
        >>> heatmap.shape
        (72, 72)
    """

    yaw_grid: np.ndarray
    pitch_grid: np.ndarray
    score_grid: np.ndarray
    resolution: int = 72
    metric_name: str = "score"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape."""
        return self.score_grid.shape

    @property
    def yaw_range(self) -> Tuple[float, float]:
        """Range of yaw angles in radians."""
        return float(self.yaw_grid.min()), float(self.yaw_grid.max())

    @property
    def pitch_range(self) -> Tuple[float, float]:
        """Range of pitch angles in radians."""
        return float(self.pitch_grid.min()), float(self.pitch_grid.max())

    @property
    def score_range(self) -> Tuple[float, float]:
        """Range of scores."""
        return float(np.nanmin(self.score_grid)), float(np.nanmax(self.score_grid))

    def to_rgb(
        self,
        colormap: str = "magma",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convert scores to RGB image.

        Args:
            colormap: Colormap name
            vmin: Minimum for normalization
            vmax: Maximum for normalization

        Returns:
            HxWx3 array of RGB values
        """
        return apply_colormap(
            self.score_grid,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            as_uint8=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'yaw_grid': self.yaw_grid.tolist(),
            'pitch_grid': self.pitch_grid.tolist(),
            'score_grid': self.score_grid.tolist(),
            'resolution': self.resolution,
            'metric_name': self.metric_name,
            'metadata': self.metadata,
        }

    def save_npz(self, path: Union[str, Path]) -> None:
        """Save to compressed NumPy file."""
        np.savez_compressed(
            path,
            yaw_grid=self.yaw_grid,
            pitch_grid=self.pitch_grid,
            score_grid=self.score_grid,
            resolution=np.array([self.resolution]),
            metric_name=np.array([self.metric_name]),
        )

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "SphericalHeatmap":
        """Load from compressed NumPy file."""
        data = np.load(path)
        return cls(
            yaw_grid=data['yaw_grid'],
            pitch_grid=data['pitch_grid'],
            score_grid=data['score_grid'],
            resolution=int(data['resolution'][0]),
            metric_name=str(data['metric_name'][0]),
        )

    def get_score_at(self, yaw: float, pitch: float) -> float:
        """
        Get interpolated score at specific angle.

        Uses bilinear interpolation.

        Args:
            yaw: Yaw angle in radians
            pitch: Pitch angle in radians

        Returns:
            Interpolated score
        """
        # Normalize yaw to grid range
        yaw_min, yaw_max = self.yaw_range
        pitch_min, pitch_max = self.pitch_range

        yaw_norm = (yaw - yaw_min) / (yaw_max - yaw_min + 1e-8)
        pitch_norm = (pitch - pitch_min) / (pitch_max - pitch_min + 1e-8)

        # Map to grid indices
        h, w = self.shape
        j = yaw_norm * (w - 1)
        i = pitch_norm * (h - 1)

        # Bilinear interpolation
        i0 = int(np.floor(i))
        i1 = min(i0 + 1, h - 1)
        j0 = int(np.floor(j))
        j1 = min(j0 + 1, w - 1)

        di = i - i0
        dj = j - j0

        score = (
            self.score_grid[i0, j0] * (1 - di) * (1 - dj) +
            self.score_grid[i0, j1] * (1 - di) * dj +
            self.score_grid[i1, j0] * di * (1 - dj) +
            self.score_grid[i1, j1] * di * dj
        )

        return float(score)


class HeatmapGenerator:
    """
    Generate spherical heatmaps from sparse angle-score data.

    Uses kernel regression for interpolation on S².

    Args:
        resolution: Grid resolution
        bandwidth: Kernel bandwidth for interpolation
        use_loocv: Use leave-one-out cross-validation for bandwidth

    Example:
        >>> generator = HeatmapGenerator(resolution=72, bandwidth=0.5)
        >>> generator.add_score(yaw=0, pitch=np.pi/2, score=0.8)
        >>> generator.add_score(yaw=np.pi, pitch=np.pi/2, score=0.6)
        >>> heatmap = generator.generate()
    """

    def __init__(
        self,
        resolution: int = 72,
        bandwidth: float = 0.5,
        use_loocv: bool = False,
    ):
        self.resolution = resolution
        self.bandwidth = bandwidth
        self.use_loocv = use_loocv

        # Storage for raw data
        self._yaws: List[float] = []
        self._pitches: List[float] = []
        self._scores: List[float] = []
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
            self._yaws.append(float(yaw))
            self._pitches.append(float(pitch))
            self._scores.append(float(score))

    def add_scores(
        self,
        yaws: np.ndarray,
        pitches: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        """
        Add multiple scores at once.

        Args:
            yaws: Array of yaw angles
            pitches: Array of pitch angles
            scores: Array of scores
        """
        for yaw, pitch, score in zip(yaws, pitches, scores):
            self.add_score(yaw, pitch, score)

    def set_metric_name(self, name: str) -> None:
        """Set the metric name for the heatmap."""
        self._metric_name = name

    @property
    def n_samples(self) -> int:
        """Number of samples added."""
        return len(self._scores)

    def clear(self) -> None:
        """Clear all stored data."""
        self._yaws.clear()
        self._pitches.clear()
        self._scores.clear()

    def generate(
        self,
        yaw_range: Tuple[float, float] = (0, 2 * np.pi),
        pitch_range: Tuple[float, float] = (0, np.pi),
    ) -> SphericalHeatmap:
        """
        Generate interpolated heatmap.

        Args:
            yaw_range: Range of yaw angles (min, max) in radians
            pitch_range: Range of pitch angles (min, max) in radians

        Returns:
            SphericalHeatmap with interpolated scores
        """
        if self.n_samples < 3:
            # Not enough data, return uniform grid
            yaws = np.linspace(yaw_range[0], yaw_range[1], self.resolution)
            pitches = np.linspace(pitch_range[0], pitch_range[1], self.resolution)
            yaw_grid, pitch_grid = np.meshgrid(yaws, pitches)

            default_score = np.mean(self._scores) if self._scores else 0.5
            score_grid = np.full_like(yaw_grid, default_score)

            return SphericalHeatmap(
                yaw_grid=yaw_grid,
                pitch_grid=pitch_grid,
                score_grid=score_grid,
                resolution=self.resolution,
                metric_name=self._metric_name,
                metadata={'n_samples': self.n_samples, 'interpolated': False},
            )

        # Create grid
        yaws = np.linspace(yaw_range[0], yaw_range[1], self.resolution)
        pitches = np.linspace(pitch_range[0], pitch_range[1], self.resolution)
        yaw_grid, pitch_grid = np.meshgrid(yaws, pitches)

        # Convert data to arrays
        data_yaws = np.array(self._yaws)
        data_pitches = np.array(self._pitches)
        data_scores = np.array(self._scores)

        # Optimize bandwidth if requested
        bandwidth = self.bandwidth
        if self.use_loocv and self.n_samples > 10:
            bandwidth = self._optimize_bandwidth(
                data_yaws, data_pitches, data_scores
            )

        # Interpolate using kernel regression
        score_grid = self._kernel_regression(
            yaw_grid.ravel(),
            pitch_grid.ravel(),
            data_yaws,
            data_pitches,
            data_scores,
            bandwidth,
        ).reshape(yaw_grid.shape)

        return SphericalHeatmap(
            yaw_grid=yaw_grid,
            pitch_grid=pitch_grid,
            score_grid=score_grid,
            resolution=self.resolution,
            metric_name=self._metric_name,
            metadata={
                'n_samples': self.n_samples,
                'bandwidth': bandwidth,
                'interpolated': True,
            },
        )

    def _kernel_regression(
        self,
        query_yaws: np.ndarray,
        query_pitches: np.ndarray,
        data_yaws: np.ndarray,
        data_pitches: np.ndarray,
        data_scores: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        """
        Nadaraya-Watson kernel regression on S².

        Uses Gaussian RBF kernel based on angular distance.
        """
        n_queries = len(query_yaws)
        n_data = len(data_yaws)
        results = np.zeros(n_queries)

        # Process in batches for memory efficiency
        batch_size = 1000
        for i in range(0, n_queries, batch_size):
            batch_end = min(i + batch_size, n_queries)
            batch_yaws = query_yaws[i:batch_end]
            batch_pitches = query_pitches[i:batch_end]

            # Compute angular distances using spherical geometry
            # cos(d) = sin(p1)*sin(p2) + cos(p1)*cos(p2)*cos(y1-y2)
            cos_dist = (
                np.sin(batch_pitches[:, np.newaxis]) * np.sin(data_pitches[np.newaxis, :]) +
                np.cos(batch_pitches[:, np.newaxis]) * np.cos(data_pitches[np.newaxis, :]) *
                np.cos(batch_yaws[:, np.newaxis] - data_yaws[np.newaxis, :])
            )
            cos_dist = np.clip(cos_dist, -1, 1)
            angular_dist = np.arccos(cos_dist)

            # Gaussian kernel
            weights = np.exp(-angular_dist ** 2 / (2 * bandwidth ** 2))

            # Normalize weights
            weight_sums = weights.sum(axis=1, keepdims=True)
            weight_sums = np.maximum(weight_sums, 1e-10)
            weights = weights / weight_sums

            # Weighted average
            results[i:batch_end] = (weights * data_scores[np.newaxis, :]).sum(axis=1)

        return results

    def _optimize_bandwidth(
        self,
        yaws: np.ndarray,
        pitches: np.ndarray,
        scores: np.ndarray,
        bandwidths: Optional[np.ndarray] = None,
    ) -> float:
        """
        Optimize bandwidth using leave-one-out cross-validation.

        Returns bandwidth that minimizes prediction error.
        """
        if bandwidths is None:
            bandwidths = np.linspace(0.1, 2.0, 20)

        n = len(scores)
        best_bandwidth = self.bandwidth
        best_error = float('inf')

        for bw in bandwidths:
            error = 0.0
            for i in range(n):
                # Leave one out
                mask = np.arange(n) != i
                pred = self._kernel_regression(
                    np.array([yaws[i]]),
                    np.array([pitches[i]]),
                    yaws[mask],
                    pitches[mask],
                    scores[mask],
                    bw,
                )[0]
                error += (pred - scores[i]) ** 2

            error /= n
            if error < best_error:
                best_error = error
                best_bandwidth = bw

        return best_bandwidth


def create_2d_heatmap(
    heatmap: SphericalHeatmap,
    colormap: str = "magma",
    figsize: Tuple[int, int] = (10, 5),
    title: Optional[str] = None,
) -> np.ndarray:
    """
    Create 2D rectangular heatmap image.

    Args:
        heatmap: SphericalHeatmap data
        colormap: Colormap name
        figsize: Figure size in inches (width, height)
        title: Optional title

    Returns:
        RGB image array

    Example:
        >>> img = create_2d_heatmap(heatmap, colormap="viridis")
        >>> img.shape  # (height, width, 3)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=figsize)

        # Convert to degrees for display
        yaw_deg = np.degrees(heatmap.yaw_grid)
        pitch_deg = np.degrees(heatmap.pitch_grid)

        im = ax.pcolormesh(
            yaw_deg, pitch_deg,
            heatmap.score_grid,
            cmap=colormap,
            shading='auto',
        )

        ax.set_xlabel('Yaw (degrees)')
        ax.set_ylabel('Pitch (degrees)')

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{heatmap.metric_name} Heatmap')

        cbar = plt.colorbar(im, ax=ax, label=heatmap.metric_name)

        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img

    except ImportError:
        # Fallback without matplotlib
        return heatmap.to_rgb(colormap=colormap)


def create_polar_heatmap(
    heatmap: SphericalHeatmap,
    colormap: str = "magma",
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
) -> np.ndarray:
    """
    Create polar projection heatmap image.

    Useful for head-on viewing angle visualization.

    Args:
        heatmap: SphericalHeatmap data
        colormap: Colormap name
        figsize: Figure size in inches
        title: Optional title

    Returns:
        RGB image array
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        # Use yaw as theta, pitch as radial
        im = ax.pcolormesh(
            heatmap.yaw_grid,
            np.degrees(heatmap.pitch_grid),
            heatmap.score_grid,
            cmap=colormap,
            shading='auto',
        )

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{heatmap.metric_name} (Polar)')

        plt.colorbar(im, ax=ax, label=heatmap.metric_name, pad=0.1)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img

    except ImportError:
        return heatmap.to_rgb(colormap=colormap)


def create_mollweide_projection(
    heatmap: SphericalHeatmap,
    colormap: str = "magma",
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
) -> np.ndarray:
    """
    Create Mollweide (equal-area) projection heatmap.

    Best for visualizing the full sphere without polar distortion.

    Args:
        heatmap: SphericalHeatmap data
        colormap: Colormap name
        figsize: Figure size in inches
        title: Optional title

    Returns:
        RGB image array
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='mollweide'))

        # Convert to longitude/latitude for Mollweide
        # yaw -> longitude (-pi to pi), pitch -> latitude (-pi/2 to pi/2)
        lon = heatmap.yaw_grid - np.pi  # Shift to [-pi, pi]
        lat = np.pi / 2 - heatmap.pitch_grid  # Convert pitch to latitude

        im = ax.pcolormesh(
            lon, lat,
            heatmap.score_grid,
            cmap=colormap,
            shading='auto',
        )

        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{heatmap.metric_name} (Mollweide Projection)')

        plt.colorbar(im, ax=ax, label=heatmap.metric_name, orientation='horizontal', pad=0.05)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img

    except ImportError:
        return heatmap.to_rgb(colormap=colormap)
