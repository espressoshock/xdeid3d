"""
Aggregate evaluation mode.

Evaluates multiple samples and computes population-level statistics.
"""

from typing import Any, Dict, List, Optional, Sequence
import time
import numpy as np

from xdeid3d.evaluation.data import (
    EvaluationSample,
    EvaluationResult,
    AggregatedResults,
    EvaluationStatus,
)
from xdeid3d.evaluation.providers import SampleProvider
from xdeid3d.evaluation.modes.base import EvaluationMode, MetricSuite

__all__ = ["AggregateMode"]


class AggregateMode(EvaluationMode):
    """
    Aggregate evaluation mode.

    Evaluates samples and computes population-level statistics
    including mean, std, percentiles, and distributions.

    Features:
    - Running statistics (no need to store all samples)
    - Confidence intervals
    - Outlier detection
    - Stratified analysis by metadata

    Args:
        metric_suite: Metrics to evaluate
        compute_confidence: Compute confidence intervals
        confidence_level: Confidence level (e.g., 0.95)
        detect_outliers: Flag outlier samples

    Example:
        >>> mode = AggregateMode()
        >>> results = mode.evaluate_provider(provider)
        >>> print(results.summary())
    """

    def __init__(
        self,
        metric_suite: Optional[MetricSuite] = None,
        compute_confidence: bool = True,
        confidence_level: float = 0.95,
        detect_outliers: bool = True,
        outlier_threshold: float = 3.0,
        verbose: bool = True,
    ):
        super().__init__(metric_suite=metric_suite, verbose=verbose)
        self.compute_confidence = compute_confidence
        self.confidence_level = confidence_level
        self.detect_outliers = detect_outliers
        self.outlier_threshold = outlier_threshold

        # Running statistics
        self._running_stats: Dict[str, Dict[str, float]] = {}

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a single sample and update running statistics."""
        start_time = time.perf_counter()

        result = EvaluationResult(
            sample_id=sample.sample_id,
            frame_index=sample.frame_index,
            yaw=sample.yaw,
            pitch=sample.pitch,
        )

        # Load images if needed
        original = sample.original
        anonymized = sample.anonymized

        if isinstance(original, str):
            sample.load_images()
            original = sample.original
            anonymized = sample.anonymized

        if original is None or anonymized is None:
            result.add_error("Failed to load images")
            return result

        # Initialize metrics if needed
        self.metric_suite.initialize_all()

        # Evaluate all metrics
        try:
            result.metrics = self.metric_suite.evaluate(
                original, anonymized, **kwargs
            )
            result.status = EvaluationStatus.COMPLETED

            # Update running statistics
            self._update_running_stats(result.metrics)

        except Exception as e:
            result.add_error(str(e))

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _update_running_stats(self, metrics: Dict[str, float]) -> None:
        """Update running statistics with new metric values."""
        for name, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                continue

            if name not in self._running_stats:
                self._running_stats[name] = {
                    'n': 0,
                    'mean': 0.0,
                    'M2': 0.0,  # For Welford's algorithm
                    'min': float('inf'),
                    'max': float('-inf'),
                }

            stats = self._running_stats[name]
            stats['n'] += 1
            delta = value - stats['mean']
            stats['mean'] += delta / stats['n']
            delta2 = value - stats['mean']
            stats['M2'] += delta * delta2
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)

    def get_running_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current running statistics."""
        result = {}
        for name, stats in self._running_stats.items():
            n = stats['n']
            if n < 2:
                var = 0.0
            else:
                var = stats['M2'] / (n - 1)

            result[name] = {
                'mean': stats['mean'],
                'std': np.sqrt(var),
                'min': stats['min'],
                'max': stats['max'],
                'count': n,
            }
        return result

    def evaluate_provider(
        self,
        provider: SampleProvider,
        **kwargs: Any,
    ) -> AggregatedResults:
        """
        Evaluate all samples with enhanced aggregation.

        Returns:
            AggregatedResults with additional statistics
        """
        # Reset running stats
        self._running_stats = {}

        # Use parent implementation for basic evaluation
        aggregated = super().evaluate_provider(provider, **kwargs)

        # Enhance with additional analysis
        if self.compute_confidence:
            aggregated = self._add_confidence_intervals(aggregated)

        if self.detect_outliers:
            aggregated = self._detect_outliers(aggregated)

        return aggregated

    def _add_confidence_intervals(
        self,
        results: AggregatedResults
    ) -> AggregatedResults:
        """Add confidence intervals to results."""
        from scipy import stats as scipy_stats

        for name, stats in results.metric_stats.items():
            n = stats.get('count', 0)
            if n < 3:
                continue

            mean = stats['mean']
            std = stats['std']

            # Compute t-based confidence interval
            se = std / np.sqrt(n)
            t_crit = scipy_stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se

            stats[f'ci_{int(self.confidence_level*100)}_low'] = float(ci_low)
            stats[f'ci_{int(self.confidence_level*100)}_high'] = float(ci_high)

        return results

    def _detect_outliers(
        self,
        results: AggregatedResults
    ) -> AggregatedResults:
        """Detect and flag outlier samples."""
        outliers: Dict[str, List[str]] = {}

        for name in results.metric_stats:
            stats = results.metric_stats[name]
            mean = stats['mean']
            std = stats['std']

            if std < 1e-10:
                continue

            outliers[name] = []
            for r in results.per_sample_results:
                if r.status != EvaluationStatus.COMPLETED:
                    continue
                value = r.metrics.get(name)
                if value is None:
                    continue

                z_score = abs(value - mean) / std
                if z_score > self.outlier_threshold:
                    outliers[name].append(r.sample_id)

            stats['outlier_count'] = len(outliers[name])

        # Store outlier info in metadata
        if not hasattr(results, 'metadata'):
            results.metadata = {}
        results.metadata = {'outliers': outliers}

        return results

    def stratified_analysis(
        self,
        results: AggregatedResults,
        stratify_by: str,
    ) -> Dict[str, AggregatedResults]:
        """
        Perform stratified analysis by a metadata field.

        Args:
            results: Aggregated results to stratify
            stratify_by: Metadata field to stratify by (e.g., 'yaw', 'pitch')

        Returns:
            Dictionary mapping stratum values to AggregatedResults
        """
        strata: Dict[Any, List[EvaluationResult]] = {}

        for r in results.per_sample_results:
            # Get stratification value from metadata or direct attribute
            if stratify_by in ['yaw', 'pitch', 'frame_index']:
                value = getattr(r, stratify_by, None)
            else:
                value = r.raw_data.get(stratify_by)

            if value is None:
                continue

            # Quantize continuous values
            if isinstance(value, float):
                value = round(value, 1)

            if value not in strata:
                strata[value] = []
            strata[value].append(r)

        return {
            str(k): AggregatedResults.from_results(v)
            for k, v in sorted(strata.items())
        }

    def clear_results(self) -> None:
        """Clear stored results and running statistics."""
        super().clear_results()
        self._running_stats = {}
