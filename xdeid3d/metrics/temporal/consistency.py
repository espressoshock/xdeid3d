"""
Temporal consistency metrics for video anonymization.

These metrics measure how stable and consistent the anonymization
is across consecutive video frames.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from xdeid3d.metrics.base import BaseMetric, MetricCategory, MetricDirection, MetricResult
from xdeid3d.metrics.registry import MetricRegistry

__all__ = [
    "TemporalIdentityConsistency",
    "TemporalVisualSmoothness",
    "FrameToFrameDistance",
]


@MetricRegistry.register("temporal_identity_consistency")
@MetricRegistry.register("tic")
class TemporalIdentityConsistency(BaseMetric):
    """
    Temporal Identity Consistency (TIC) metric.

    Measures how stable the anonymized identity is across consecutive frames.
    A good anonymizer should produce consistent anonymized identities
    (low variation between frames).

    Formula: TIC = 1 - mean(cosine_distance(emb_t, emb_{t+1}))

    Higher values indicate more consistent anonymization.
    Range: [0, 1] where 1 = perfectly consistent.

    Args:
        model_name: Face recognition model for embeddings
        use_cache: Enable embedding caching

    Example:
        >>> metric = TemporalIdentityConsistency()
        >>> result = metric.compute_sequence(anonymized_frames)
        >>> print(f"TIC: {result.value:.4f}")
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        use_cache: bool = True,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="temporal_identity_consistency",
            category=MetricCategory.TEMPORAL,
            direction=MetricDirection.HIGHER_BETTER,
            device=device,
        )
        self.model_name = model_name
        self.use_cache = use_cache
        self._extractor = None
        self._cache = None

    def initialize(self) -> None:
        """Initialize embedding extractor."""
        from xdeid3d.metrics.identity.embeddings import (
            InsightFaceExtractor,
            EmbeddingCache,
        )

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
    ) -> float:
        """
        For single frame pair, return 1.0 (no temporal aspect).
        Use compute_sequence for proper temporal analysis.
        """
        return 1.0

    def compute_sequence(
        self,
        frames: Sequence[np.ndarray],
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute TIC over a sequence of frames.

        Args:
            frames: Sequence of anonymized video frames

        Returns:
            MetricResult with TIC value and per-frame distances
        """
        import time
        from xdeid3d.metrics.identity.embeddings import cosine_distance

        self._ensure_initialized()

        if len(frames) < 2:
            return MetricResult(
                name=self._name,
                value=1.0,
                category=self._category,
                direction=self._direction,
                metadata={"warning": "Need at least 2 frames for TIC"},
            )

        start_time = time.perf_counter()

        # Extract embeddings for all frames
        embeddings = []
        detection_mask = []
        for frame in frames:
            frame = self._validate_input(frame)
            emb = self._get_embedding(frame)
            embeddings.append(emb)
            detection_mask.append(emb is not None)

        # Compute consecutive frame distances
        distances = []
        for i in range(len(embeddings) - 1):
            if embeddings[i] is not None and embeddings[i + 1] is not None:
                dist = cosine_distance(embeddings[i], embeddings[i + 1])
                distances.append(dist)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if not distances:
            return MetricResult(
                name=self._name,
                value=0.0,
                category=self._category,
                direction=self._direction,
                metadata={"warning": "No consecutive face detections"},
                processing_time_ms=elapsed_ms,
            )

        # TIC = 1 - mean distance (higher = more consistent)
        mean_distance = np.mean(distances)
        tic_value = 1.0 - mean_distance

        return MetricResult(
            name=self._name,
            value=float(tic_value),
            category=self._category,
            direction=self._direction,
            raw_values=np.array(distances),
            metadata={
                "mean_distance": float(mean_distance),
                "std_distance": float(np.std(distances)),
                "detection_rate": sum(detection_mask) / len(detection_mask),
                "n_frames": len(frames),
            },
            processing_time_ms=elapsed_ms,
        )


@MetricRegistry.register("temporal_visual_smoothness")
@MetricRegistry.register("tvs")
class TemporalVisualSmoothness(BaseMetric):
    """
    Temporal Visual Smoothness (TVS) metric.

    Measures visual smoothness between consecutive frames, detecting
    jerkiness or flickering in the anonymized output.

    Uses frame-to-frame pixel differences normalized by the frame content.
    Lower frame differences indicate smoother video.

    Higher TVS = smoother video.

    Example:
        >>> metric = TemporalVisualSmoothness()
        >>> result = metric.compute_sequence(anonymized_frames)
        >>> print(f"TVS: {result.value:.4f}")
    """

    def __init__(
        self,
        normalize: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            name="temporal_visual_smoothness",
            category=MetricCategory.TEMPORAL,
            direction=MetricDirection.HIGHER_BETTER,
        )
        self.normalize = normalize

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Single frame pair returns 1.0."""
        return 1.0

    def compute_sequence(
        self,
        frames: Sequence[np.ndarray],
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute TVS over a sequence of frames.

        Args:
            frames: Sequence of anonymized video frames

        Returns:
            MetricResult with TVS value
        """
        import time

        self._ensure_initialized()

        if len(frames) < 2:
            return MetricResult(
                name=self._name,
                value=1.0,
                category=self._category,
                direction=self._direction,
            )

        start_time = time.perf_counter()

        # Compute frame-to-frame differences
        differences = []
        for i in range(len(frames) - 1):
            frame1 = self._validate_input(frames[i]).astype(np.float32)
            frame2 = self._validate_input(frames[i + 1]).astype(np.float32)

            # Mean absolute difference
            diff = np.mean(np.abs(frame1 - frame2))

            if self.normalize:
                # Normalize by average frame intensity
                avg_intensity = (np.mean(frame1) + np.mean(frame2)) / 2
                diff = diff / (avg_intensity + 1e-8)

            differences.append(diff)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # TVS = 1 - normalized_diff (higher = smoother)
        mean_diff = np.mean(differences)
        tvs_value = 1.0 - min(mean_diff, 1.0)  # Clamp to [0, 1]

        return MetricResult(
            name=self._name,
            value=float(tvs_value),
            category=self._category,
            direction=self._direction,
            raw_values=np.array(differences),
            metadata={
                "mean_difference": float(mean_diff),
                "max_difference": float(np.max(differences)),
                "jerkiness_score": float(np.std(differences)),
                "n_frames": len(frames),
            },
            processing_time_ms=elapsed_ms,
        )


@MetricRegistry.register("frame_to_frame_distance")
class FrameToFrameDistance(BaseMetric):
    """
    Frame-to-frame distance metric.

    Computes various distance metrics between original and anonymized
    frame sequences to measure temporal coherence of anonymization.

    Example:
        >>> metric = FrameToFrameDistance()
        >>> result = metric.compute_batch(original_frames, anonymized_frames)
    """

    def __init__(
        self,
        distance_type: str = "mse",
        **kwargs: Any,
    ):
        super().__init__(
            name="frame_to_frame_distance",
            category=MetricCategory.TEMPORAL,
            direction=MetricDirection.LOWER_BETTER,
        )
        self.distance_type = distance_type

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute distance between single frame pair."""
        orig = original.astype(np.float32) / 255.0
        anon = anonymized.astype(np.float32) / 255.0

        if self.distance_type == "mse":
            return float(np.mean((orig - anon) ** 2))
        elif self.distance_type == "mae":
            return float(np.mean(np.abs(orig - anon)))
        elif self.distance_type == "max":
            return float(np.max(np.abs(orig - anon)))
        else:
            return float(np.mean((orig - anon) ** 2))
