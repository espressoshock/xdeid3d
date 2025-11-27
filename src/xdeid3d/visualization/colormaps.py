"""
Colormap utilities for visualization.

Provides color mapping functions for converting scores to RGB colors
for heatmap visualization on 3D meshes.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

__all__ = [
    "get_colormap",
    "apply_colormap",
    "normalize_scores",
    "ColorMapper",
    "COLORMAPS",
]

# Pre-defined colormap data (256 entries each)
# These are sampled from matplotlib colormaps for standalone use

def _generate_viridis() -> np.ndarray:
    """Generate viridis colormap data."""
    # Viridis colormap - perceptually uniform
    t = np.linspace(0, 1, 256)
    # Simplified viridis approximation
    r = 0.267004 + t * (0.993248 - 0.267004) * (1 - np.exp(-3 * t))
    g = 0.004874 + t * (0.906157 - 0.004874)
    b = 0.329415 + (0.143936 - 0.329415) * t + 0.5 * np.sin(np.pi * t) * 0.3
    return np.column_stack([
        np.clip(r * 0.4 + 0.1, 0, 1),
        np.clip(g * 0.8 + 0.1, 0, 1),
        np.clip(b * 0.6 + 0.2, 0, 1),
    ])


def _generate_magma() -> np.ndarray:
    """Generate magma colormap data."""
    t = np.linspace(0, 1, 256)
    # Magma approximation
    r = t ** 0.5
    g = t ** 2
    b = np.sin(np.pi * t * 0.8) * 0.8 + 0.2 * t
    return np.column_stack([
        np.clip(r, 0, 1),
        np.clip(g, 0, 1),
        np.clip(b * 0.7, 0, 1),
    ])


def _generate_plasma() -> np.ndarray:
    """Generate plasma colormap data."""
    t = np.linspace(0, 1, 256)
    r = 0.05 + 0.95 * t ** 0.7
    g = t * (1 - t) * 2
    b = 0.9 - 0.85 * t
    return np.column_stack([
        np.clip(r, 0, 1),
        np.clip(g * 1.5, 0, 1),
        np.clip(b, 0, 1),
    ])


def _generate_inferno() -> np.ndarray:
    """Generate inferno colormap data."""
    t = np.linspace(0, 1, 256)
    r = t ** 0.4
    g = t ** 1.5 * 0.8
    b = np.sin(np.pi * t * 0.5) * 0.6
    return np.column_stack([
        np.clip(r, 0, 1),
        np.clip(g, 0, 1),
        np.clip(b, 0, 1),
    ])


def _generate_hot() -> np.ndarray:
    """Generate hot colormap data."""
    t = np.linspace(0, 1, 256)
    r = np.clip(t * 3, 0, 1)
    g = np.clip((t - 0.33) * 3, 0, 1)
    b = np.clip((t - 0.67) * 3, 0, 1)
    return np.column_stack([r, g, b])


def _generate_coolwarm() -> np.ndarray:
    """Generate coolwarm (diverging) colormap data."""
    t = np.linspace(0, 1, 256)
    # Blue to white to red
    r = np.where(t < 0.5, 0.2 + t, 1.0)
    g = np.where(t < 0.5, 0.2 + t * 1.6, 1.0 - (t - 0.5) * 1.6)
    b = np.where(t < 0.5, 1.0, 1.0 - (t - 0.5) * 1.6)
    return np.column_stack([
        np.clip(r, 0, 1),
        np.clip(g, 0, 1),
        np.clip(b, 0, 1),
    ])


def _generate_turbo() -> np.ndarray:
    """Generate turbo colormap data (improved jet)."""
    t = np.linspace(0, 1, 256)
    # Rainbow-like but more perceptually uniform
    r = np.sin(np.pi * (t + 0.0)) ** 2
    g = np.sin(np.pi * (t + 0.33)) ** 2
    b = np.sin(np.pi * (t + 0.67)) ** 2
    return np.column_stack([r, g, b])


def _generate_grayscale() -> np.ndarray:
    """Generate grayscale colormap data."""
    t = np.linspace(0, 1, 256)
    return np.column_stack([t, t, t])


def _generate_red_green() -> np.ndarray:
    """Generate red-green diverging colormap (good vs bad)."""
    t = np.linspace(0, 1, 256)
    # Red (bad) -> Yellow -> Green (good)
    r = np.where(t < 0.5, 1.0, 1.0 - (t - 0.5) * 2)
    g = np.where(t < 0.5, t * 2, 1.0)
    b = np.zeros_like(t)
    return np.column_stack([
        np.clip(r, 0, 1),
        np.clip(g, 0, 1),
        b,
    ])


# Registry of built-in colormaps
COLORMAPS = {
    "viridis": _generate_viridis,
    "magma": _generate_magma,
    "plasma": _generate_plasma,
    "inferno": _generate_inferno,
    "hot": _generate_hot,
    "coolwarm": _generate_coolwarm,
    "turbo": _generate_turbo,
    "grayscale": _generate_grayscale,
    "gray": _generate_grayscale,
    "red_green": _generate_red_green,
    "performance": _generate_red_green,  # Alias for metric visualization
}


def get_colormap(name: str = "viridis") -> np.ndarray:
    """
    Get colormap data by name.

    Args:
        name: Colormap name. Options: viridis, magma, plasma, inferno,
              hot, coolwarm, turbo, grayscale, red_green, performance

    Returns:
        Nx3 array of RGB values in [0, 1]

    Raises:
        ValueError: If colormap name is unknown

    Example:
        >>> cmap = get_colormap("magma")
        >>> cmap.shape
        (256, 3)
    """
    name = name.lower()

    if name in COLORMAPS:
        return COLORMAPS[name]()

    # Try matplotlib if available
    try:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap(name)
        t = np.linspace(0, 1, 256)
        return cmap(t)[:, :3]
    except (ImportError, ValueError):
        pass

    raise ValueError(
        f"Unknown colormap: {name}. "
        f"Available: {list(COLORMAPS.keys())}"
    )


def normalize_scores(
    scores: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip: bool = True,
) -> np.ndarray:
    """
    Normalize scores to [0, 1] range.

    Args:
        scores: Array of scores
        vmin: Minimum value (default: scores.min())
        vmax: Maximum value (default: scores.max())
        clip: Whether to clip values outside [0, 1]

    Returns:
        Normalized scores in [0, 1]

    Example:
        >>> scores = np.array([0.2, 0.5, 0.8])
        >>> normalize_scores(scores, vmin=0, vmax=1)
        array([0.2, 0.5, 0.8])
    """
    scores = np.asarray(scores, dtype=np.float64)

    if vmin is None:
        vmin = np.nanmin(scores)
    if vmax is None:
        vmax = np.nanmax(scores)

    if vmax - vmin < 1e-10:
        return np.full_like(scores, 0.5)

    normalized = (scores - vmin) / (vmax - vmin)

    if clip:
        normalized = np.clip(normalized, 0, 1)

    return normalized


def apply_colormap(
    scores: np.ndarray,
    colormap: Union[str, np.ndarray] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    as_uint8: bool = False,
) -> np.ndarray:
    """
    Apply colormap to scores.

    Args:
        scores: Array of scores (any shape)
        colormap: Colormap name or Nx3 array
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        as_uint8: Return as uint8 (0-255) instead of float (0-1)

    Returns:
        Array of RGB colors with shape (*scores.shape, 3)

    Example:
        >>> scores = np.array([0.0, 0.5, 1.0])
        >>> colors = apply_colormap(scores, "hot")
        >>> colors.shape
        (3, 3)
    """
    scores = np.asarray(scores)
    original_shape = scores.shape
    scores_flat = scores.ravel()

    # Get colormap
    if isinstance(colormap, str):
        cmap_data = get_colormap(colormap)
    else:
        cmap_data = np.asarray(colormap)

    # Normalize scores
    normalized = normalize_scores(scores_flat, vmin, vmax)

    # Map to colormap indices
    indices = (normalized * (len(cmap_data) - 1)).astype(int)
    indices = np.clip(indices, 0, len(cmap_data) - 1)

    # Look up colors
    colors = cmap_data[indices]

    # Reshape to original shape + color channels
    colors = colors.reshape(*original_shape, 3)

    if as_uint8:
        colors = (colors * 255).astype(np.uint8)

    return colors


class ColorMapper:
    """
    Reusable color mapper for consistent colormap application.

    Useful when applying the same colormap multiple times
    with fixed normalization bounds.

    Args:
        colormap: Colormap name or data
        vmin: Fixed minimum for normalization
        vmax: Fixed maximum for normalization
        reverse: Reverse the colormap direction

    Example:
        >>> mapper = ColorMapper("magma", vmin=0, vmax=1)
        >>> colors = mapper(np.array([0.2, 0.5, 0.8]))
        >>> mapper.to_uint8(np.array([0.5]))
        array([[...]], dtype=uint8)
    """

    def __init__(
        self,
        colormap: Union[str, np.ndarray] = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        reverse: bool = False,
    ):
        self.colormap_name = colormap if isinstance(colormap, str) else "custom"
        self._cmap_data = get_colormap(colormap) if isinstance(colormap, str) else colormap

        if reverse:
            self._cmap_data = self._cmap_data[::-1].copy()

        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        """Apply colormap to scores."""
        return apply_colormap(
            scores,
            colormap=self._cmap_data,
            vmin=self.vmin,
            vmax=self.vmax,
            as_uint8=False,
        )

    def to_uint8(self, scores: np.ndarray) -> np.ndarray:
        """Apply colormap and return as uint8."""
        return apply_colormap(
            scores,
            colormap=self._cmap_data,
            vmin=self.vmin,
            vmax=self.vmax,
            as_uint8=True,
        )

    def get_colorbar_data(
        self,
        height: int = 256,
        width: int = 32,
    ) -> np.ndarray:
        """
        Generate colorbar image data.

        Args:
            height: Height in pixels
            width: Width in pixels

        Returns:
            HxWx3 array of RGB values (uint8)
        """
        gradient = np.linspace(1, 0, height)[:, np.newaxis]
        gradient = np.repeat(gradient, width, axis=1)
        return self.to_uint8(gradient)

    @property
    def n_colors(self) -> int:
        """Number of colors in the colormap."""
        return len(self._cmap_data)
