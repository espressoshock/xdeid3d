"""
Regional saliency metrics for anonymization explainability.

These metrics identify which face regions contribute most to
identity change or preservation during anonymization.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from xdeid3d.metrics.base import BaseMetric, MetricCategory, MetricDirection, MetricResult
from xdeid3d.metrics.registry import MetricRegistry

__all__ = [
    "RegionalSaliencyMetric",
    "DifferenceMapMetric",
    "IdentityContributionMap",
]


@MetricRegistry.register("regional_saliency")
class RegionalSaliencyMetric(BaseMetric):
    """
    Computes regional saliency map showing identity-relevant face regions.

    Uses perturbation-based or gradient-based analysis to identify
    which image regions contribute most to identity change.

    Args:
        grid_size: Size of the analysis grid (e.g., 8 for 8x8)
        method: Saliency computation method ('occlusion', 'gradient', 'difference')

    Example:
        >>> metric = RegionalSaliencyMetric(grid_size=8)
        >>> result = metric.compute(original, anonymized)
        >>> saliency_map = result.metadata['saliency_map']
    """

    def __init__(
        self,
        grid_size: int = 8,
        method: str = "difference",
        **kwargs: Any,
    ):
        super().__init__(
            name="regional_saliency",
            category=MetricCategory.EXPLAINABILITY,
            direction=MetricDirection.HIGHER_BETTER,
        )
        self.grid_size = grid_size
        self.method = method

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute saliency and return mean saliency value."""
        saliency_map = self._compute_saliency_map(original, anonymized)
        return float(np.mean(saliency_map))

    def _compute_saliency_map(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-region saliency map.

        Returns:
            2D array of shape (grid_size, grid_size) with saliency values
        """
        h, w = original.shape[:2]
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size

        saliency = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = i * cell_h
                y2 = min((i + 1) * cell_h, h)
                x1 = j * cell_w
                x2 = min((j + 1) * cell_w, w)

                orig_region = original[y1:y2, x1:x2].astype(np.float32)
                anon_region = anonymized[y1:y2, x1:x2].astype(np.float32)

                if self.method == "difference":
                    # Simple pixel-wise difference
                    diff = np.abs(orig_region - anon_region)
                    saliency[i, j] = np.mean(diff) / 255.0

                elif self.method == "structural":
                    # Structural difference (luminance + contrast)
                    orig_mean = np.mean(orig_region)
                    anon_mean = np.mean(anon_region)
                    orig_std = np.std(orig_region)
                    anon_std = np.std(anon_region)

                    luminance_diff = abs(orig_mean - anon_mean) / 255.0
                    contrast_diff = abs(orig_std - anon_std) / 128.0

                    saliency[i, j] = (luminance_diff + contrast_diff) / 2

                else:
                    # Default to normalized difference
                    diff = np.abs(orig_region - anon_region)
                    saliency[i, j] = np.mean(diff) / 255.0

        return saliency

    def compute(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute regional saliency.

        Returns:
            MetricResult with saliency_map in metadata
        """
        import time
        start_time = time.perf_counter()

        original = self._validate_input(original)
        anonymized = self._validate_input(anonymized)

        saliency_map = self._compute_saliency_map(original, anonymized)
        mean_saliency = float(np.mean(saliency_map))

        # Identify high-saliency regions
        threshold = np.percentile(saliency_map, 75)
        high_saliency_mask = saliency_map > threshold
        high_saliency_coords = np.argwhere(high_saliency_mask)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MetricResult(
            name=self._name,
            value=mean_saliency,
            category=self._category,
            direction=self._direction,
            raw_values=saliency_map.flatten(),
            metadata={
                'saliency_map': saliency_map.tolist(),
                'grid_size': self.grid_size,
                'method': self.method,
                'max_saliency': float(np.max(saliency_map)),
                'min_saliency': float(np.min(saliency_map)),
                'std_saliency': float(np.std(saliency_map)),
                'high_saliency_regions': high_saliency_coords.tolist(),
                'high_saliency_fraction': float(np.mean(high_saliency_mask)),
            },
            processing_time_ms=elapsed_ms,
        )


@MetricRegistry.register("difference_map")
class DifferenceMapMetric(BaseMetric):
    """
    Computes pixel-wise difference heatmap between original and anonymized.

    Useful for visualizing where changes occur and their magnitude.

    Args:
        normalize: Normalize output to [0, 1] range
        colormap: Colormap for visualization (stored in metadata)

    Example:
        >>> metric = DifferenceMapMetric()
        >>> result = metric.compute(original, anonymized)
        >>> diff_map = result.metadata['difference_map']
    """

    def __init__(
        self,
        normalize: bool = True,
        colormap: str = "jet",
        **kwargs: Any,
    ):
        super().__init__(
            name="difference_map",
            category=MetricCategory.EXPLAINABILITY,
            direction=MetricDirection.LOWER_BETTER,
        )
        self.normalize = normalize
        self.colormap = colormap

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute mean difference."""
        diff = np.abs(original.astype(np.float32) - anonymized.astype(np.float32))
        if self.normalize:
            diff = diff / 255.0
        return float(np.mean(diff))

    def compute(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute difference map.

        Returns:
            MetricResult with difference_map in metadata
        """
        import time
        start_time = time.perf_counter()

        original = self._validate_input(original)
        anonymized = self._validate_input(anonymized)

        # Compute difference
        diff = np.abs(original.astype(np.float32) - anonymized.astype(np.float32))

        # Convert to grayscale if color
        if diff.ndim == 3:
            diff_gray = np.mean(diff, axis=2)
        else:
            diff_gray = diff

        if self.normalize:
            diff_gray = diff_gray / 255.0

        mean_diff = float(np.mean(diff_gray))

        # Compute statistics
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MetricResult(
            name=self._name,
            value=mean_diff,
            category=self._category,
            direction=self._direction,
            raw_values=diff_gray.flatten()[:1000],  # Sample for memory
            metadata={
                'difference_map_shape': list(diff_gray.shape),
                'max_difference': float(np.max(diff_gray)),
                'min_difference': float(np.min(diff_gray)),
                'std_difference': float(np.std(diff_gray)),
                'percentile_90': float(np.percentile(diff_gray, 90)),
                'percentile_95': float(np.percentile(diff_gray, 95)),
                'colormap': self.colormap,
                'normalized': self.normalize,
            },
            processing_time_ms=elapsed_ms,
        )

    def get_heatmap(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
    ) -> np.ndarray:
        """
        Get difference as a color heatmap image.

        Returns:
            RGB heatmap array of shape (H, W, 3)
        """
        original = self._validate_input(original)
        anonymized = self._validate_input(anonymized)

        diff = np.abs(original.astype(np.float32) - anonymized.astype(np.float32))

        if diff.ndim == 3:
            diff_gray = np.mean(diff, axis=2)
        else:
            diff_gray = diff

        # Normalize to [0, 255]
        diff_normalized = (diff_gray / diff_gray.max() * 255).astype(np.uint8)

        # Apply colormap
        try:
            import cv2
            colormap_id = getattr(cv2, f'COLORMAP_{self.colormap.upper()}', cv2.COLORMAP_JET)
            heatmap = cv2.applyColorMap(diff_normalized, colormap_id)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        except ImportError:
            # Fallback: grayscale to RGB
            heatmap = np.stack([diff_normalized] * 3, axis=-1)

        return heatmap


@MetricRegistry.register("identity_contribution_map")
class IdentityContributionMap(BaseMetric):
    """
    Maps which regions contribute most to identity recognition.

    Uses occlusion-based perturbation to measure how much each
    region contributes to the overall identity embedding.

    Args:
        grid_size: Size of the occlusion grid
        occlusion_value: Pixel value for occlusion (0=black, 128=gray)

    Example:
        >>> metric = IdentityContributionMap(grid_size=8)
        >>> result = metric.compute(original, anonymized)
        >>> contribution_map = result.metadata['contribution_map']
    """

    def __init__(
        self,
        grid_size: int = 8,
        occlusion_value: int = 128,
        **kwargs: Any,
    ):
        super().__init__(
            name="identity_contribution_map",
            category=MetricCategory.EXPLAINABILITY,
            direction=MetricDirection.HIGHER_BETTER,
        )
        self.grid_size = grid_size
        self.occlusion_value = occlusion_value
        self._extractor = None
        self._base_embedding = None

    def initialize(self) -> None:
        """Initialize embedding extractor."""
        from xdeid3d.metrics.identity.embeddings import InsightFaceExtractor

        self._extractor = InsightFaceExtractor()
        self._initialized = True

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute mean contribution."""
        contribution_map = self._compute_contribution_map(original, anonymized)
        return float(np.mean(contribution_map))

    def _compute_contribution_map(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
    ) -> np.ndarray:
        """
        Compute contribution map using occlusion analysis.

        For each region, measure how much identity distance changes
        when that region is occluded.
        """
        from xdeid3d.metrics.identity.embeddings import cosine_distance

        self._ensure_initialized()

        h, w = original.shape[:2]
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size

        # Get base embeddings
        orig_emb = self._extractor.extract(original)
        anon_emb = self._extractor.extract(anonymized)

        if orig_emb is None or anon_emb is None:
            return np.zeros((self.grid_size, self.grid_size))

        base_distance = cosine_distance(orig_emb, anon_emb)
        contribution = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = i * cell_h
                y2 = min((i + 1) * cell_h, h)
                x1 = j * cell_w
                x2 = min((j + 1) * cell_w, w)

                # Occlude region in anonymized image
                anon_occluded = anonymized.copy()
                anon_occluded[y1:y2, x1:x2] = self.occlusion_value

                # Get embedding with occlusion
                occluded_emb = self._extractor.extract(anon_occluded)

                if occluded_emb is not None:
                    occluded_distance = cosine_distance(orig_emb, occluded_emb)
                    # Contribution = how much distance changes when region is occluded
                    contribution[i, j] = abs(occluded_distance - base_distance)
                else:
                    contribution[i, j] = 0.0

        # Normalize
        if contribution.max() > 0:
            contribution = contribution / contribution.max()

        return contribution

    def compute(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute identity contribution map.

        Returns:
            MetricResult with contribution_map in metadata
        """
        import time
        start_time = time.perf_counter()

        original = self._validate_input(original)
        anonymized = self._validate_input(anonymized)

        contribution_map = self._compute_contribution_map(original, anonymized)
        mean_contribution = float(np.mean(contribution_map))

        # Identify high-contribution regions
        threshold = np.percentile(contribution_map, 75)
        high_contrib_mask = contribution_map > threshold
        high_contrib_coords = np.argwhere(high_contrib_mask)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MetricResult(
            name=self._name,
            value=mean_contribution,
            category=self._category,
            direction=self._direction,
            raw_values=contribution_map.flatten(),
            metadata={
                'contribution_map': contribution_map.tolist(),
                'grid_size': self.grid_size,
                'max_contribution': float(np.max(contribution_map)),
                'min_contribution': float(np.min(contribution_map)),
                'std_contribution': float(np.std(contribution_map)),
                'high_contribution_regions': high_contrib_coords.tolist(),
                'occlusion_value': self.occlusion_value,
            },
            processing_time_ms=elapsed_ms,
        )
