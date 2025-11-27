"""
Base evaluation mode interface.

Evaluation modes define how metrics are computed and aggregated
for different use cases (single sample, aggregate, 3D spherical).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, Union
import time
import numpy as np

from xdeid3d.evaluation.data import (
    EvaluationSample,
    EvaluationResult,
    AggregatedResults,
    EvaluationStatus,
)
from xdeid3d.evaluation.providers import SampleProvider
from xdeid3d.metrics.base import BaseMetric, MetricResult

__all__ = [
    "EvaluationMode",
    "MetricSuite",
]


class MetricSuite:
    """
    Collection of metrics to evaluate together.

    Groups related metrics for efficient batch evaluation.

    Args:
        metrics: List of metric instances or (name, metric) tuples
        name: Suite name

    Example:
        >>> suite = MetricSuite([
        ...     ArcFaceCosineDistance(),
        ...     SSIMMetric(),
        ...     PSNRMetric(),
        ... ], name="standard")
        >>> results = suite.evaluate(original, anonymized)
    """

    def __init__(
        self,
        metrics: Union[List[BaseMetric], List[tuple]],
        name: str = "default",
    ):
        self.name = name
        self._metrics: Dict[str, BaseMetric] = {}

        for item in metrics:
            if isinstance(item, tuple):
                metric_name, metric = item
            else:
                metric_name = item.name
                metric = item
            self._metrics[metric_name] = metric

    def add_metric(self, metric: BaseMetric, name: Optional[str] = None) -> None:
        """Add a metric to the suite."""
        name = name or metric.name
        self._metrics[name] = metric

    def remove_metric(self, name: str) -> None:
        """Remove a metric by name."""
        self._metrics.pop(name, None)

    @property
    def metric_names(self) -> List[str]:
        """Get list of metric names."""
        return list(self._metrics.keys())

    def initialize_all(self) -> None:
        """Initialize all metrics."""
        for metric in self._metrics.values():
            if hasattr(metric, 'initialize') and not getattr(metric, '_initialized', False):
                metric.initialize()

    def evaluate(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Evaluate all metrics on a sample.

        Returns:
            Dictionary of metric_name -> value
        """
        results = {}
        for name, metric in self._metrics.items():
            try:
                result = metric.compute(original, anonymized, **kwargs)
                results[name] = result.value
            except Exception as e:
                results[name] = float('nan')
        return results

    def evaluate_detailed(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, MetricResult]:
        """
        Evaluate all metrics and return full MetricResult objects.

        Returns:
            Dictionary of metric_name -> MetricResult
        """
        results = {}
        for name, metric in self._metrics.items():
            try:
                results[name] = metric.compute(original, anonymized, **kwargs)
            except Exception as e:
                results[name] = MetricResult(
                    name=name,
                    value=float('nan'),
                    category=metric.category,
                    direction=metric.direction,
                    metadata={'error': str(e)},
                )
        return results

    @classmethod
    def standard(cls) -> "MetricSuite":
        """Create standard evaluation suite."""
        from xdeid3d.metrics.identity.arcface import ArcFaceCosineDistance
        from xdeid3d.metrics.quality.psnr import PSNRMetric
        from xdeid3d.metrics.quality.ssim import SSIMMetric

        return cls([
            ArcFaceCosineDistance(),
            PSNRMetric(),
            SSIMMetric(),
        ], name="standard")

    @classmethod
    def full(cls) -> "MetricSuite":
        """Create comprehensive evaluation suite."""
        from xdeid3d.metrics.identity.arcface import ArcFaceCosineDistance, IdentityChangeMetric
        from xdeid3d.metrics.quality.psnr import PSNRMetric
        from xdeid3d.metrics.quality.ssim import SSIMMetric

        metrics = [
            ArcFaceCosineDistance(),
            IdentityChangeMetric(),
            PSNRMetric(),
            SSIMMetric(),
        ]

        # Try to add LPIPS if available
        try:
            from xdeid3d.metrics.quality.lpips import LPIPSMetric
            metrics.append(LPIPSMetric())
        except ImportError:
            pass

        return cls(metrics, name="full")


class EvaluationMode(ABC):
    """
    Abstract base class for evaluation modes.

    Evaluation modes control how samples are processed and
    how results are aggregated.

    Subclasses:
        - SingleSampleMode: Evaluate individual samples
        - AggregateMode: Aggregate over multiple samples
        - SphericalMode: 3D spherical evaluation with pose tracking
    """

    def __init__(
        self,
        metric_suite: Optional[MetricSuite] = None,
        verbose: bool = True,
    ):
        self.metric_suite = metric_suite or MetricSuite.standard()
        self.verbose = verbose
        self._results: List[EvaluationResult] = []

    @abstractmethod
    def evaluate_sample(
        self,
        sample: EvaluationSample,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a single sample."""
        pass

    def evaluate_provider(
        self,
        provider: SampleProvider,
        **kwargs: Any,
    ) -> AggregatedResults:
        """
        Evaluate all samples from a provider.

        Args:
            provider: Sample provider to iterate over
            **kwargs: Additional arguments passed to evaluate_sample

        Returns:
            AggregatedResults with statistics
        """
        self._results = []
        total = len(provider)

        if self.verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(provider, total=total, desc=f"Evaluating ({self.__class__.__name__})")
            except ImportError:
                iterator = provider
        else:
            iterator = provider

        for sample in iterator:
            result = self.evaluate_sample(sample, **kwargs)
            self._results.append(result)

        return AggregatedResults.from_results(self._results)

    def evaluate_batch(
        self,
        samples: Sequence[EvaluationSample],
        **kwargs: Any,
    ) -> AggregatedResults:
        """
        Evaluate a batch of samples.

        Args:
            samples: Sequence of samples
            **kwargs: Additional arguments

        Returns:
            AggregatedResults
        """
        self._results = []

        for sample in samples:
            result = self.evaluate_sample(sample, **kwargs)
            self._results.append(result)

        return AggregatedResults.from_results(self._results)

    def get_results(self) -> List[EvaluationResult]:
        """Get all evaluation results."""
        return self._results

    def clear_results(self) -> None:
        """Clear stored results."""
        self._results = []
