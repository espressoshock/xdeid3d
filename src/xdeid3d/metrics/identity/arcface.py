"""
ArcFace-based identity metrics.

This module provides identity metrics using ArcFace embeddings,
including cosine distance for measuring identity change.
"""

from typing import Any, Dict, Optional
import numpy as np

from xdeid3d.metrics.base import BaseMetric, MetricCategory, MetricDirection
from xdeid3d.metrics.registry import MetricRegistry
from xdeid3d.metrics.identity.embeddings import (
    InsightFaceExtractor,
    EmbeddingCache,
    EmbeddingExtractor,
    cosine_distance,
)

__all__ = [
    "ArcFaceCosineDistance",
    "IdentityChangeMetric",
]


@MetricRegistry.register("arcface_cosine_distance")
@MetricRegistry.register("identity_distance")
class ArcFaceCosineDistance(BaseMetric):
    """
    ArcFace cosine distance metric.

    Measures identity change between original and anonymized faces
    using ArcFace embeddings with cosine distance.

    Higher values indicate more identity change (better anonymization).

    Args:
        model_name: InsightFace model name
        use_cache: Enable embedding caching
        threshold: Identity verification threshold

    Example:
        >>> metric = ArcFaceCosineDistance()
        >>> result = metric.compute(original, anonymized)
        >>> print(f"Identity distance: {result.value:.3f}")
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        use_cache: bool = True,
        threshold: float = 0.5,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="arcface_cosine_distance",
            category=MetricCategory.IDENTITY,
            direction=MetricDirection.HIGHER_BETTER,  # Higher = more anonymization
            device=device,
        )
        self.model_name = model_name
        self.use_cache = use_cache
        self.threshold = threshold
        self._extractor: Optional[InsightFaceExtractor] = None
        self._cache: Optional[EmbeddingCache] = None

    def initialize(self) -> None:
        """Initialize the extractor and cache."""
        device = self._device or "cpu"
        self._extractor = InsightFaceExtractor(
            model_name=self.model_name,
            device=device,
        )
        if self.use_cache:
            self._cache = EmbeddingCache(max_size=10000)
        self._initialized = True

    def _get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get embedding with optional caching."""
        if self._cache is not None:
            return self._cache.get_or_compute(image, self._extractor)
        return self._extractor.extract(image)

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute identity distance between original and anonymized.

        Returns dict with:
        - value: Cosine distance (0 = same, 2 = opposite)
        - verified: Whether identities match (< threshold)
        - original_detected: Whether face detected in original
        - anonymized_detected: Whether face detected in anonymized
        """
        emb_original = self._get_embedding(original)
        emb_anonymized = self._get_embedding(anonymized)

        result = {
            "original_detected": emb_original is not None,
            "anonymized_detected": emb_anonymized is not None,
        }

        if emb_original is None or emb_anonymized is None:
            # No face detected in one or both images
            result["value"] = 0.0 if emb_original is None else 1.0
            result["verified"] = False
            return result

        distance = cosine_distance(emb_original, emb_anonymized)
        result["value"] = distance
        result["verified"] = distance < self.threshold

        return result


@MetricRegistry.register("identity_change")
class IdentityChangeMetric(BaseMetric):
    """
    Binary identity change metric.

    Returns 1.0 if identity changed (distance > threshold), 0.0 otherwise.
    Useful for computing anonymization success rate.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        threshold: float = 0.5,
        use_cache: bool = True,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="identity_change",
            category=MetricCategory.IDENTITY,
            direction=MetricDirection.HIGHER_BETTER,
            device=device,
        )
        self.model_name = model_name
        self.threshold = threshold
        self.use_cache = use_cache
        self._extractor: Optional[InsightFaceExtractor] = None
        self._cache: Optional[EmbeddingCache] = None

    def initialize(self) -> None:
        device = self._device or "cpu"
        self._extractor = InsightFaceExtractor(
            model_name=self.model_name,
            device=device,
        )
        if self.use_cache:
            self._cache = EmbeddingCache(max_size=10000)
        self._initialized = True

    def _get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self._cache is not None:
            return self._cache.get_or_compute(image, self._extractor)
        return self._extractor.extract(image)

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, float]:
        emb_original = self._get_embedding(original)
        emb_anonymized = self._get_embedding(anonymized)

        if emb_original is None or emb_anonymized is None:
            # Conservative: if can't detect face, assume identity unchanged
            return {"value": 0.0, "face_detected": False}

        distance = cosine_distance(emb_original, emb_anonymized)
        changed = 1.0 if distance >= self.threshold else 0.0

        return {
            "value": changed,
            "distance": distance,
            "face_detected": True,
        }
