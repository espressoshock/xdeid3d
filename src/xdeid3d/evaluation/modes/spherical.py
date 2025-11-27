"""
Spherical (3D) evaluation mode.

Evaluates anonymization across viewing angles on S² for
3D heatmap visualization and pose-robustness analysis.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
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
from xdeid3d.metrics.explainability.viewpoint import (
    SphericalScoreInterpolator,
    AngleBasedEvaluator,
)

__all__ = ["SphericalMode"]


class SphericalMode(EvaluationMode):
    """
    Spherical (3D) evaluation mode.

    Evaluates anonymization performance across viewing angles,
    enabling 3D heatmap visualizations and pose-robustness analysis.

    Features:
    - Per-angle metric tracking
    - Spherical interpolation for dense heatmaps
    - Pose robustness scoring
    - Weak angle identification

    Args:
        metric_suite: Metrics to evaluate
        interpolation_bandwidth: Bandwidth for spherical interpolation
        grid_resolution: Resolution for output heatmap grid
        primary_metric: Which metric to use for primary analysis

    Example:
        >>> mode = SphericalMode(primary_metric="arcface_cosine_distance")
        >>> results = mode.evaluate_provider(provider)
        >>> heatmap = mode.get_spherical_heatmap()
    """

    def __init__(
        self,
        metric_suite: Optional[MetricSuite] = None,
        interpolation_bandwidth: float = 0.5,
        grid_resolution: int = 72,
        primary_metric: str = "arcface_cosine_distance",
        verbose: bool = True,
    ):
        super().__init__(metric_suite=metric_suite, verbose=verbose)
        self.interpolation_bandwidth = interpolation_bandwidth
        self.grid_resolution = grid_resolution
        self.primary_metric = primary_metric

        # Per-metric angle-based evaluators
        self._evaluators: Dict[str, AngleBasedEvaluator] = {}
        self._interpolators: Dict[str, SphericalScoreInterpolator] = {}

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate a single sample with pose tracking.

        Requires sample to have yaw and pitch set.
        """
        start_time = time.perf_counter()

        result = EvaluationResult(
            sample_id=sample.sample_id,
            frame_index=sample.frame_index,
            yaw=sample.yaw,
            pitch=sample.pitch,
        )

        # Check for pose information
        if sample.yaw is None or sample.pitch is None:
            result.add_warning("Sample missing pose information")
            # Try to extract from metadata
            if 'yaw' in sample.metadata:
                sample.yaw = sample.metadata['yaw']
            if 'pitch' in sample.metadata:
                sample.pitch = sample.metadata['pitch']

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

            # Store in angle-based evaluators
            if sample.yaw is not None and sample.pitch is not None:
                for name, value in result.metrics.items():
                    if np.isnan(value) or np.isinf(value):
                        continue

                    if name not in self._evaluators:
                        self._evaluators[name] = AngleBasedEvaluator(name)

                    self._evaluators[name].add_score(
                        sample.yaw, sample.pitch, value
                    )

        except Exception as e:
            result.add_error(str(e))

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def evaluate_provider(
        self,
        provider: SampleProvider,
        **kwargs: Any,
    ) -> AggregatedResults:
        """
        Evaluate all samples and build spherical models.

        Returns:
            AggregatedResults with spherical analysis
        """
        # Reset evaluators
        self._evaluators = {}
        self._interpolators = {}

        # Use parent implementation for basic evaluation
        aggregated = super().evaluate_provider(provider, **kwargs)

        # Build interpolators for each metric
        self._build_interpolators()

        # Add spherical analysis to results
        aggregated = self._add_spherical_analysis(aggregated)

        return aggregated

    def _build_interpolators(self) -> None:
        """Build spherical interpolators from collected data."""
        for name, evaluator in self._evaluators.items():
            scores = evaluator.get_all_scores()
            if len(scores) < 3:
                continue

            interpolator = SphericalScoreInterpolator(
                bandwidth=self.interpolation_bandwidth,
                use_loocv=len(scores) > 20,
            )
            interpolator.fit(
                yaws=np.array([s['yaw'] for s in scores]),
                pitches=np.array([s['pitch'] for s in scores]),
                scores=np.array([s['score'] for s in scores]),
            )
            self._interpolators[name] = interpolator

    def _add_spherical_analysis(
        self,
        results: AggregatedResults
    ) -> AggregatedResults:
        """Add spherical analysis to aggregated results."""
        spherical_stats = {}

        for name, evaluator in self._evaluators.items():
            stats = evaluator.compute_statistics()

            # Compute pose robustness
            if stats['count'] > 1:
                robustness = 1.0 - min(stats['std'] / 0.5, 1.0)
            else:
                robustness = 1.0

            # Identify weak angles
            weak_angles = self._identify_weak_angles(name)

            spherical_stats[name] = {
                **stats,
                'pose_robustness': float(robustness),
                'weak_angles': weak_angles,
            }

        # Store in results
        if not hasattr(results, 'metadata') or results.metadata is None:
            results.metadata = {}
        results.metadata['spherical_analysis'] = spherical_stats

        return results

    def _identify_weak_angles(
        self,
        metric_name: str,
        threshold_std: float = 1.5,
    ) -> List[Dict[str, float]]:
        """Identify viewing angles with poor performance."""
        if metric_name not in self._evaluators:
            return []

        evaluator = self._evaluators[metric_name]
        stats = evaluator.compute_statistics()
        scores = evaluator.get_all_scores()

        if stats['count'] < 3 or stats['std'] < 1e-10:
            return []

        mean = stats['mean']
        std = stats['std']
        threshold = mean - threshold_std * std

        weak = []
        for s in scores:
            if s['score'] < threshold:
                weak.append({
                    'yaw': float(s['yaw']),
                    'pitch': float(s['pitch']),
                    'yaw_deg': float(np.degrees(s['yaw'])),
                    'pitch_deg': float(np.degrees(s['pitch'])),
                    'score': float(s['score']),
                    'z_score': float((s['score'] - mean) / std),
                })

        # Sort by score (worst first)
        return sorted(weak, key=lambda x: x['score'])[:20]

    def get_spherical_heatmap(
        self,
        metric_name: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get interpolated heatmap on the viewing sphere.

        Args:
            metric_name: Metric to get heatmap for (default: primary_metric)
            resolution: Grid resolution (default: grid_resolution)

        Returns:
            Tuple of (yaw_grid, pitch_grid, score_grid) in radians
        """
        metric_name = metric_name or self.primary_metric
        resolution = resolution or self.grid_resolution

        if metric_name not in self._interpolators:
            # Return empty grid
            yaws = np.linspace(0, 2 * np.pi, resolution)
            pitches = np.linspace(0, np.pi, resolution)
            yaw_grid, pitch_grid = np.meshgrid(yaws, pitches)
            return yaw_grid, pitch_grid, np.full_like(yaw_grid, 0.5)

        return self._interpolators[metric_name].predict_grid(
            resolution=resolution
        )

    def get_score_at_angle(
        self,
        yaw: float,
        pitch: float,
        metric_name: Optional[str] = None,
    ) -> float:
        """
        Get interpolated score at a specific viewing angle.

        Args:
            yaw: Yaw angle in radians
            pitch: Pitch angle in radians
            metric_name: Metric to query (default: primary_metric)

        Returns:
            Interpolated score
        """
        metric_name = metric_name or self.primary_metric

        if metric_name not in self._interpolators:
            return 0.5

        return self._interpolators[metric_name].predict(yaw, pitch)

    def get_vertex_colors(
        self,
        vertices: np.ndarray,
        metric_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get colors for mesh vertices based on metric scores.

        Args:
            vertices: Nx3 array of vertex positions
            metric_name: Metric to use for coloring

        Returns:
            Nx1 array of normalized scores for vertex coloring
        """
        metric_name = metric_name or self.primary_metric

        if metric_name not in self._interpolators:
            return np.full(len(vertices), 0.5)

        interpolator = self._interpolators[metric_name]
        scores = np.zeros(len(vertices))

        for i, vertex in enumerate(vertices):
            # Convert vertex to spherical coordinates
            norm = np.linalg.norm(vertex) + 1e-8
            direction = vertex / norm

            yaw = np.arctan2(direction[2], direction[0]) + np.pi
            pitch = np.arccos(np.clip(direction[1], -1, 1))

            scores[i] = interpolator.predict(yaw, pitch)

        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def get_pose_robustness(self, metric_name: Optional[str] = None) -> float:
        """
        Get pose robustness score for a metric.

        Returns:
            Robustness score in [0, 1], higher = more consistent across poses
        """
        metric_name = metric_name or self.primary_metric

        if metric_name not in self._evaluators:
            return 1.0

        stats = self._evaluators[metric_name].compute_statistics()
        if stats['count'] < 2:
            return 1.0

        # Robustness = 1 - normalized_std
        return 1.0 - min(stats['std'] / 0.5, 1.0)

    def clear_results(self) -> None:
        """Clear stored results and spherical models."""
        super().clear_results()
        self._evaluators = {}
        self._interpolators = {}
