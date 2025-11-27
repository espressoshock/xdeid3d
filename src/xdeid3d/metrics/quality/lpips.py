"""
Learned Perceptual Image Patch Similarity (LPIPS) metric.
"""

from typing import Any, Dict, Optional
import numpy as np

from xdeid3d.metrics.base import BaseMetric, MetricCategory, MetricDirection
from xdeid3d.metrics.registry import MetricRegistry

__all__ = ["LPIPSMetric"]


@MetricRegistry.register("lpips")
class LPIPSMetric(BaseMetric):
    """
    Learned Perceptual Image Patch Similarity metric.

    LPIPS measures perceptual similarity using deep features from
    pretrained networks (AlexNet, VGG, or SqueezeNet).

    Lower values indicate more perceptually similar images.
    Range: [0, ~1], where 0 = identical.

    Args:
        net: Network backbone ('alex', 'vgg', 'squeeze')
        device: Device for computation ('cuda' or 'cpu')

    Example:
        >>> metric = LPIPSMetric(net='alex')
        >>> result = metric.compute(original, anonymized)
        >>> print(f"LPIPS: {result.value:.4f}")
    """

    def __init__(
        self,
        net: str = "alex",
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="lpips",
            category=MetricCategory.QUALITY,
            direction=MetricDirection.LOWER_BETTER,  # Lower = more similar
            device=device,
        )
        self.net = net
        self._model = None

    def initialize(self) -> None:
        """Initialize LPIPS model."""
        try:
            import torch
            import lpips

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = lpips.LPIPS(net=self.net).to(device)
            self._device = device
            self._initialized = True

        except ImportError:
            raise ImportError(
                "lpips is required for LPIPS metric. "
                "Install with: pip install lpips"
            )

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute LPIPS distance between images."""
        import torch

        # Convert to torch tensors
        # LPIPS expects [-1, 1] range with shape (N, C, H, W)
        orig = self._preprocess(original)
        anon = self._preprocess(anonymized)

        # Compute LPIPS
        with torch.no_grad():
            distance = self._model(orig, anon)

        return float(distance.item())

    def _preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """Preprocess image for LPIPS."""
        import torch

        # Ensure RGB order (H, W, C)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Normalize to [-1, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 127.5 - 1.0
        elif image.max() > 1:
            image = image / 127.5 - 1.0
        else:
            image = image * 2 - 1.0

        # Convert to (N, C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.float().to(self._device)
