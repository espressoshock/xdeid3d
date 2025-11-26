"""
Single sample evaluation mode.

Evaluates each sample independently without cross-sample aggregation.
"""

from typing import Any, Dict, Optional
import time
import numpy as np

from xdeid3d.evaluation.data import (
    EvaluationSample,
    EvaluationResult,
    EvaluationStatus,
)
from xdeid3d.evaluation.modes.base import EvaluationMode, MetricSuite

__all__ = ["SingleSampleMode"]


class SingleSampleMode(EvaluationMode):
    """
    Single sample evaluation mode.

    Evaluates each sample independently, computing all metrics
    in the metric suite for each original-anonymized pair.

    Args:
        metric_suite: Metrics to evaluate
        fail_on_error: Stop on first error vs. continue
        include_raw_data: Include detailed metric data in results

    Example:
        >>> mode = SingleSampleMode()
        >>> result = mode.evaluate_sample(sample)
        >>> print(result.metrics)
    """

    def __init__(
        self,
        metric_suite: Optional[MetricSuite] = None,
        fail_on_error: bool = False,
        include_raw_data: bool = False,
        verbose: bool = True,
    ):
        super().__init__(metric_suite=metric_suite, verbose=verbose)
        self.fail_on_error = fail_on_error
        self.include_raw_data = include_raw_data

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate a single sample.

        Args:
            sample: Sample to evaluate
            **kwargs: Additional metric arguments

        Returns:
            EvaluationResult with all metric values
        """
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
            if self.include_raw_data:
                detailed_results = self.metric_suite.evaluate_detailed(
                    original, anonymized, **kwargs
                )
                for name, metric_result in detailed_results.items():
                    result.metrics[name] = metric_result.value
                    if metric_result.raw_values is not None:
                        result.raw_data[f"{name}_raw"] = metric_result.raw_values.tolist()
                    if metric_result.metadata:
                        result.raw_data[f"{name}_meta"] = metric_result.metadata
            else:
                result.metrics = self.metric_suite.evaluate(
                    original, anonymized, **kwargs
                )

            result.status = EvaluationStatus.COMPLETED

        except Exception as e:
            result.add_error(str(e))
            if self.fail_on_error:
                raise

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def evaluate_with_pose(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        yaw: float,
        pitch: float,
        sample_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate a single frame with pose information.

        Convenience method for direct array evaluation with pose.

        Args:
            original: Original image array
            anonymized: Anonymized image array
            yaw: Yaw angle in radians
            pitch: Pitch angle in radians
            sample_id: Optional sample ID
            **kwargs: Additional metric arguments

        Returns:
            EvaluationResult with pose metadata
        """
        sample = EvaluationSample(
            sample_id=sample_id or f"sample_{int(time.time()*1000)}",
            original=original,
            anonymized=anonymized,
            yaw=yaw,
            pitch=pitch,
        )
        return self.evaluate_sample(sample, **kwargs)
