"""
Base metric protocol and classes.

This module defines the protocol (interface) that all metrics must implement,
along with base classes and result containers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Sequence, TypeVar, Union, runtime_checkable
import numpy as np

__all__ = [
    "MetricProtocol",
    "BaseMetric",
    "MetricResult",
    "MetricCategory",
    "MetricDirection",
    "AggregationMethod",
]


class MetricCategory(str, Enum):
    """Categories for organizing metrics."""

    IDENTITY = "identity"  # Face identity preservation/change
    QUALITY = "quality"  # Visual quality (PSNR, SSIM, LPIPS)
    TEMPORAL = "temporal"  # Temporal consistency
    EXPLAINABILITY = "explainability"  # Attribution/explanation metrics


class MetricDirection(str, Enum):
    """Direction indicating whether higher or lower is better."""

    HIGHER_BETTER = "higher_better"  # Higher values are better (e.g., PSNR)
    LOWER_BETTER = "lower_better"  # Lower values are better (e.g., LPIPS)
    NEUTRAL = "neutral"  # No preference (e.g., raw measurements)


class AggregationMethod(str, Enum):
    """Methods for aggregating metric values."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    PERCENTILE_5 = "p5"
    PERCENTILE_95 = "p95"


@dataclass
class MetricResult:
    """
    Result of computing a metric.

    Attributes:
        name: Metric name
        value: Primary metric value
        category: Metric category
        direction: Whether higher or lower is better
        raw_values: Raw values before aggregation (for per-frame metrics)
        metadata: Additional metric-specific data
        processing_time_ms: Time taken to compute in milliseconds
    """

    name: str
    value: float
    category: MetricCategory = MetricCategory.QUALITY
    direction: MetricDirection = MetricDirection.HIGHER_BETTER
    raw_values: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    @property
    def is_better_higher(self) -> bool:
        """Check if higher values are better."""
        return self.direction == MetricDirection.HIGHER_BETTER

    @property
    def statistics(self) -> Dict[str, float]:
        """Compute statistics from raw values."""
        if self.raw_values is None or len(self.raw_values) == 0:
            return {"value": self.value}

        return {
            "mean": float(np.mean(self.raw_values)),
            "std": float(np.std(self.raw_values)),
            "min": float(np.min(self.raw_values)),
            "max": float(np.max(self.raw_values)),
            "median": float(np.median(self.raw_values)),
            "p5": float(np.percentile(self.raw_values, 5)),
            "p95": float(np.percentile(self.raw_values, 95)),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "category": self.category.value,
            "direction": self.direction.value,
            "statistics": self.statistics,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
        }


@runtime_checkable
class MetricProtocol(Protocol):
    """
    Protocol defining the interface for metrics.

    Any class implementing this protocol can be used as a metric
    in the X-DeID3D evaluation pipeline.
    """

    @property
    def name(self) -> str:
        """Return the metric name."""
        ...

    @property
    def category(self) -> MetricCategory:
        """Return the metric category."""
        ...

    def compute(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute the metric for a single frame.

        Args:
            original: Original image (H, W, C)
            anonymized: Anonymized image (H, W, C)
            **kwargs: Additional metric-specific parameters

        Returns:
            MetricResult containing the computed value
        """
        ...


T = TypeVar("T", bound="BaseMetric")


class BaseMetric(ABC):
    """
    Abstract base class for metrics.

    This class provides a template for implementing metrics with
    common functionality like batch processing and timing.

    Subclasses must implement:
        - _compute_single: Core metric computation for a single pair

    Example:
        >>> class PSNRMetric(BaseMetric):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="PSNR",
        ...             category=MetricCategory.QUALITY,
        ...             direction=MetricDirection.HIGHER_BETTER,
        ...         )
        ...
        ...     def _compute_single(self, original, anonymized, **kwargs):
        ...         psnr_value = compute_psnr(original, anonymized)
        ...         return psnr_value
    """

    def __init__(
        self,
        name: str,
        category: MetricCategory = MetricCategory.QUALITY,
        direction: MetricDirection = MetricDirection.HIGHER_BETTER,
        device: Optional[str] = None,
    ):
        """
        Initialize base metric.

        Args:
            name: Metric name identifier
            category: Metric category
            direction: Whether higher or lower is better
            device: Device to use (e.g., 'cuda', 'cpu')
        """
        self._name = name
        self._category = category
        self._direction = direction
        self._device = device
        self._initialized = False

    @property
    def name(self) -> str:
        """Return the metric name."""
        return self._name

    @property
    def category(self) -> MetricCategory:
        """Return the metric category."""
        return self._category

    @property
    def direction(self) -> MetricDirection:
        """Return the direction (higher/lower better)."""
        return self._direction

    @property
    def device(self) -> Optional[str]:
        """Return the device being used."""
        return self._device

    def initialize(self) -> None:
        """
        Initialize the metric (load models, etc.).

        Override this method if your metric needs initialization.
        Called automatically on first use if not called explicitly.
        """
        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure the metric is initialized."""
        if not self._initialized:
            self.initialize()

    @abstractmethod
    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> Union[float, Dict[str, float]]:
        """
        Core metric computation for a single frame pair.

        Subclasses must implement this method.

        Args:
            original: Original image (H, W, C), uint8
            anonymized: Anonymized image (H, W, C), uint8
            **kwargs: Additional parameters

        Returns:
            Metric value (float) or dict of values for multi-output metrics
        """
        pass

    def compute(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute the metric for a single frame pair.

        Args:
            original: Original image (H, W, C)
            anonymized: Anonymized image (H, W, C)
            **kwargs: Additional metric-specific parameters

        Returns:
            MetricResult containing the computed value
        """
        import time

        self._ensure_initialized()

        # Validate inputs
        original = self._validate_input(original)
        anonymized = self._validate_input(anonymized)

        # Time the computation
        start_time = time.perf_counter()
        result = self._compute_single(original, anonymized, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Handle dict result (multi-output metrics)
        if isinstance(result, dict):
            primary_value = result.get("value", result.get(self._name, 0.0))
            metadata = {k: v for k, v in result.items() if k != "value"}
        else:
            primary_value = float(result)
            metadata = {}

        return MetricResult(
            name=self._name,
            value=primary_value,
            category=self._category,
            direction=self._direction,
            metadata=metadata,
            processing_time_ms=elapsed_ms,
        )

    def compute_batch(
        self,
        original_frames: Sequence[np.ndarray],
        anonymized_frames: Sequence[np.ndarray],
        aggregate: AggregationMethod = AggregationMethod.MEAN,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute metric for a batch of frame pairs.

        Args:
            original_frames: Sequence of original images
            anonymized_frames: Sequence of anonymized images
            aggregate: Aggregation method for combining values
            **kwargs: Additional parameters

        Returns:
            MetricResult with aggregated value and raw_values
        """
        import time

        self._ensure_initialized()

        if len(original_frames) != len(anonymized_frames):
            raise ValueError("Frame sequences must have equal length")

        start_time = time.perf_counter()
        values = []

        for orig, anon in zip(original_frames, anonymized_frames):
            result = self.compute(orig, anon, **kwargs)
            values.append(result.value)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        raw_values = np.array(values)

        # Aggregate
        agg_value = self._aggregate(raw_values, aggregate)

        return MetricResult(
            name=self._name,
            value=agg_value,
            category=self._category,
            direction=self._direction,
            raw_values=raw_values,
            processing_time_ms=elapsed_ms,
        )

    def _aggregate(
        self, values: np.ndarray, method: AggregationMethod
    ) -> float:
        """Aggregate values using specified method."""
        if method == AggregationMethod.MEAN:
            return float(np.mean(values))
        elif method == AggregationMethod.MEDIAN:
            return float(np.median(values))
        elif method == AggregationMethod.MIN:
            return float(np.min(values))
        elif method == AggregationMethod.MAX:
            return float(np.max(values))
        elif method == AggregationMethod.STD:
            return float(np.std(values))
        elif method == AggregationMethod.PERCENTILE_5:
            return float(np.percentile(values, 5))
        elif method == AggregationMethod.PERCENTILE_95:
            return float(np.percentile(values, 95))
        else:
            return float(np.mean(values))

    def _validate_input(self, image: np.ndarray) -> np.ndarray:
        """Validate and normalize input image."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")

        if image.ndim == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")

        if image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]

        # Normalize to uint8
        if image.dtype in (np.float32, np.float64):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"

    def __call__(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> MetricResult:
        """Allow calling the metric directly."""
        return self.compute(original, anonymized, **kwargs)
