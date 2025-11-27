"""
Peak Signal-to-Noise Ratio (PSNR) metric.
"""

from typing import Any, Dict
import numpy as np

from xdeid3d.metrics.base import BaseMetric, MetricCategory, MetricDirection
from xdeid3d.metrics.registry import MetricRegistry

__all__ = ["PSNRMetric"]


@MetricRegistry.register("psnr")
class PSNRMetric(BaseMetric):
    """
    Peak Signal-to-Noise Ratio metric.

    PSNR measures pixel-level similarity between images.
    Higher values indicate more similar images (less distortion).

    Formula: PSNR = 10 * log10(MAX² / MSE)
    where MAX is the maximum pixel value (255 for uint8).

    Typical ranges:
    - > 40 dB: Excellent quality
    - 30-40 dB: Good quality
    - 20-30 dB: Average quality
    - < 20 dB: Poor quality

    Example:
        >>> metric = PSNRMetric()
        >>> result = metric.compute(original, anonymized)
        >>> print(f"PSNR: {result.value:.2f} dB")
    """

    def __init__(self, max_value: float = 255.0, **kwargs: Any):
        super().__init__(
            name="psnr",
            category=MetricCategory.QUALITY,
            direction=MetricDirection.HIGHER_BETTER,
        )
        self.max_value = max_value

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute PSNR between images."""
        # Convert to float for computation
        orig = original.astype(np.float64)
        anon = anonymized.astype(np.float64)

        # Compute MSE
        mse = np.mean((orig - anon) ** 2)

        if mse == 0:
            return float("inf")

        # Compute PSNR
        psnr = 10 * np.log10((self.max_value ** 2) / mse)
        return float(psnr)
