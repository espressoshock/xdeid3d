"""
Structural Similarity Index (SSIM) metric.
"""

from typing import Any, Dict, Optional
import numpy as np

from xdeid3d.metrics.base import BaseMetric, MetricCategory, MetricDirection
from xdeid3d.metrics.registry import MetricRegistry

__all__ = ["SSIMMetric"]


@MetricRegistry.register("ssim")
class SSIMMetric(BaseMetric):
    """
    Structural Similarity Index metric.

    SSIM measures structural similarity between images, considering
    luminance, contrast, and structure components.

    Range: [-1, 1], where 1 = identical images
    Typically > 0 for natural images.

    Args:
        win_size: Size of the sliding window (odd number)
        data_range: Dynamic range of the input (255 for uint8)
        use_sample_covariance: Use sample covariance (N-1 divisor)

    Example:
        >>> metric = SSIMMetric()
        >>> result = metric.compute(original, anonymized)
        >>> print(f"SSIM: {result.value:.4f}")
    """

    def __init__(
        self,
        win_size: int = 7,
        data_range: float = 255.0,
        use_sample_covariance: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            name="ssim",
            category=MetricCategory.QUALITY,
            direction=MetricDirection.HIGHER_BETTER,
        )
        self.win_size = win_size
        self.data_range = data_range
        self.use_sample_covariance = use_sample_covariance

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute SSIM between images."""
        try:
            from skimage.metrics import structural_similarity
        except ImportError:
            # Fallback to simplified implementation
            return self._compute_ssim_simple(original, anonymized)

        # Convert to grayscale for SSIM if color
        if original.ndim == 3:
            # Use channel_axis parameter
            ssim_value = structural_similarity(
                original,
                anonymized,
                win_size=self.win_size,
                data_range=self.data_range,
                channel_axis=2,
            )
        else:
            ssim_value = structural_similarity(
                original,
                anonymized,
                win_size=self.win_size,
                data_range=self.data_range,
            )

        return float(ssim_value)

    def _compute_ssim_simple(
        self, original: np.ndarray, anonymized: np.ndarray
    ) -> float:
        """Simplified SSIM computation (fallback)."""
        # Convert to grayscale if color
        if original.ndim == 3:
            original = np.mean(original, axis=2)
            anonymized = np.mean(anonymized, axis=2)

        orig = original.astype(np.float64)
        anon = anonymized.astype(np.float64)

        # Constants for stability
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        # Compute means
        mu1 = np.mean(orig)
        mu2 = np.mean(anon)

        # Compute variances and covariance
        sigma1_sq = np.var(orig)
        sigma2_sq = np.var(anon)
        sigma12 = np.mean((orig - mu1) * (anon - mu2))

        # SSIM formula
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

        return float(numerator / denominator)
