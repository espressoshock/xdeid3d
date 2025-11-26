"""
Viewpoint-dependent explainability metrics for anonymization.

These metrics quantify how anonymization performance varies with
viewing angle, enabling 3D explanations of anonymizer behavior.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from xdeid3d.metrics.base import BaseMetric, MetricCategory, MetricDirection, MetricResult
from xdeid3d.metrics.registry import MetricRegistry

__all__ = [
    "ViewpointMetric",
    "SphericalScoreInterpolator",
    "PoseRobustnessMetric",
    "AngleBasedEvaluator",
]


class AngleBasedEvaluator:
    """
    Base class for evaluators that compute scores based on viewing angles.

    This class stores metric values at specific (yaw, pitch) angles and
    provides methods for querying and interpolating scores.

    Attributes:
        metric_name: Name of the metric being evaluated
        scores: Dictionary mapping (yaw_deg, pitch_deg) to scores

    Example:
        >>> evaluator = AngleBasedEvaluator("identity_change")
        >>> evaluator.add_score(yaw=1.57, pitch=1.57, score=0.85)
        >>> evaluator.get_score(yaw=1.57, pitch=1.57)
        0.85
    """

    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.scores: Dict[Tuple[int, int], float] = {}

    def add_score(self, yaw: float, pitch: float, score: float) -> None:
        """
        Add a score for a specific viewing angle.

        Args:
            yaw: Yaw angle in radians
            pitch: Pitch angle in radians
            score: Metric score value
        """
        yaw_deg = int(np.degrees(yaw) % 360)
        pitch_deg = int(np.degrees(pitch))
        self.scores[(yaw_deg, pitch_deg)] = score

    def get_score(self, yaw: float, pitch: float) -> Optional[float]:
        """Get score for exact angle, or None if not stored."""
        yaw_deg = int(np.degrees(yaw) % 360)
        pitch_deg = int(np.degrees(pitch))
        return self.scores.get((yaw_deg, pitch_deg))

    def get_nearest_score(self, yaw: float, pitch: float) -> Tuple[float, float]:
        """
        Get score from nearest stored angle.

        Returns:
            Tuple of (score, angular_distance)
        """
        if not self.scores:
            return (0.5, float('inf'))

        yaw_deg = np.degrees(yaw) % 360
        pitch_deg = np.degrees(pitch)

        min_dist = float('inf')
        best_score = 0.5

        for (y_deg, p_deg), score in self.scores.items():
            # Angular distance with wrap-around for yaw
            yaw_dist = min(abs(y_deg - yaw_deg), 360 - abs(y_deg - yaw_deg))
            pitch_dist = abs(p_deg - pitch_deg)
            dist = np.sqrt(yaw_dist**2 + pitch_dist**2)

            if dist < min_dist:
                min_dist = dist
                best_score = score

        return (best_score, min_dist)

    def get_all_scores(self) -> List[Dict[str, float]]:
        """Get all stored scores as list of dicts."""
        return [
            {'yaw': np.radians(yaw), 'pitch': np.radians(pitch), 'score': score}
            for (yaw, pitch), score in self.scores.items()
        ]

    def compute_statistics(self) -> Dict[str, float]:
        """Compute statistics over all stored scores."""
        if not self.scores:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}

        values = list(self.scores.values())
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values),
        }


class SphericalScoreInterpolator:
    """
    Interpolates metric scores across the viewing sphere using kernel regression.

    Uses Nadaraya-Watson estimator with Gaussian RBF kernel on S² to
    produce smooth score interpolations across viewing angles.

    This is the core component for creating 3D heatmap visualizations
    where mesh vertices are colored by interpolated metric values.

    Args:
        bandwidth: Kernel bandwidth (sigma) in radians
        use_loocv: Use leave-one-out cross-validation for bandwidth selection

    Example:
        >>> interpolator = SphericalScoreInterpolator(bandwidth=0.5)
        >>> interpolator.fit(yaws, pitches, scores)
        >>> score = interpolator.predict(yaw=1.0, pitch=1.5)
    """

    def __init__(
        self,
        bandwidth: float = 0.5,
        use_loocv: bool = False,
    ):
        self.bandwidth = bandwidth
        self.use_loocv = use_loocv
        self._yaws: Optional[np.ndarray] = None
        self._pitches: Optional[np.ndarray] = None
        self._scores: Optional[np.ndarray] = None
        self._estimator = None

    def fit(
        self,
        yaws: np.ndarray,
        pitches: np.ndarray,
        scores: np.ndarray,
    ) -> "SphericalScoreInterpolator":
        """
        Fit the interpolator to observed scores.

        Args:
            yaws: Array of yaw angles in radians
            pitches: Array of pitch angles in radians
            scores: Array of metric scores

        Returns:
            self for method chaining
        """
        self._yaws = np.asarray(yaws)
        self._pitches = np.asarray(pitches)
        self._scores = np.asarray(scores)

        # Optionally use LOOCV for bandwidth selection
        if self.use_loocv and len(scores) > 3:
            try:
                from xdeid3d.core.regression.bandwidth import LOOCVBandwidthSelector
                from xdeid3d.core.regression.kernels import GaussianRBFKernel

                selector = LOOCVBandwidthSelector(
                    kernel=GaussianRBFKernel(),
                    bandwidth_range=(0.1, 2.0),
                    n_candidates=20,
                )
                # Convert to theta/phi for spherical distance
                points = np.column_stack([self._yaws, self._pitches])
                self.bandwidth = selector.select(points, self._scores)
            except ImportError:
                pass  # Use default bandwidth

        return self

    def predict(self, yaw: float, pitch: float) -> float:
        """
        Predict score at a given viewing angle.

        Uses Gaussian-weighted average of observed scores,
        where weights decrease with angular distance.

        Args:
            yaw: Query yaw angle in radians
            pitch: Query pitch angle in radians

        Returns:
            Interpolated score value
        """
        if self._scores is None or len(self._scores) == 0:
            return 0.5

        # Compute angular distances using great-circle distance approximation
        # For small angles on a sphere
        delta_yaw = self._yaws - yaw
        delta_pitch = self._pitches - pitch

        # Wrap yaw difference
        delta_yaw = np.minimum(
            np.abs(delta_yaw),
            2 * np.pi - np.abs(delta_yaw)
        )

        # Approximate angular distance
        angular_dist = np.sqrt(
            (delta_yaw * np.sin(pitch))**2 + delta_pitch**2
        )

        # Gaussian kernel weights
        weights = np.exp(-angular_dist**2 / (2 * self.bandwidth**2))

        total_weight = np.sum(weights)
        if total_weight < 1e-10:
            return float(np.mean(self._scores))

        return float(np.sum(self._scores * weights) / total_weight)

    def predict_grid(
        self,
        yaw_range: Tuple[float, float] = (0, 2 * np.pi),
        pitch_range: Tuple[float, float] = (0, np.pi),
        resolution: int = 36,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict scores on a regular grid of viewing angles.

        Args:
            yaw_range: (min, max) yaw in radians
            pitch_range: (min, max) pitch in radians
            resolution: Number of grid points per dimension

        Returns:
            Tuple of (yaw_grid, pitch_grid, score_grid)
        """
        yaws = np.linspace(yaw_range[0], yaw_range[1], resolution)
        pitches = np.linspace(pitch_range[0], pitch_range[1], resolution)

        yaw_grid, pitch_grid = np.meshgrid(yaws, pitches)
        score_grid = np.zeros_like(yaw_grid)

        for i in range(resolution):
            for j in range(resolution):
                score_grid[i, j] = self.predict(yaw_grid[i, j], pitch_grid[i, j])

        return yaw_grid, pitch_grid, score_grid


@MetricRegistry.register("viewpoint_metric")
class ViewpointMetric(BaseMetric):
    """
    Metric that tracks performance variation across viewing angles.

    Wraps another metric and records its values at each viewing angle,
    enabling analysis of how anonymization quality varies with pose.

    Args:
        base_metric: The underlying metric to evaluate
        interpolator_bandwidth: Bandwidth for spherical interpolation

    Example:
        >>> from xdeid3d.metrics.identity import ArcFaceCosineDistance
        >>> base_metric = ArcFaceCosineDistance()
        >>> viewpoint_metric = ViewpointMetric(base_metric)
        >>>
        >>> for frame, (yaw, pitch) in zip(frames, poses):
        ...     result = viewpoint_metric.compute_with_pose(
        ...         original, anonymized, yaw=yaw, pitch=pitch
        ...     )
        >>>
        >>> # Get interpolated heatmap
        >>> heatmap = viewpoint_metric.get_spherical_heatmap(resolution=72)
    """

    def __init__(
        self,
        base_metric: Optional[BaseMetric] = None,
        interpolator_bandwidth: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(
            name="viewpoint_metric",
            category=MetricCategory.EXPLAINABILITY,
            direction=MetricDirection.HIGHER_BETTER,
        )
        self.base_metric = base_metric
        self.interpolator_bandwidth = interpolator_bandwidth
        self._evaluator = AngleBasedEvaluator(
            base_metric.name if base_metric else "unknown"
        )
        self._interpolator: Optional[SphericalScoreInterpolator] = None

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute using base metric."""
        if self.base_metric is None:
            return 0.5

        result = self.base_metric.compute(original, anonymized, **kwargs)
        return result.value

    def compute_with_pose(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        yaw: float,
        pitch: float,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute metric at a specific viewing angle.

        Args:
            original: Original image
            anonymized: Anonymized image
            yaw: Yaw angle in radians
            pitch: Pitch angle in radians

        Returns:
            MetricResult with pose information in metadata
        """
        value = self._compute_single(original, anonymized, **kwargs)

        # Store in evaluator
        self._evaluator.add_score(yaw, pitch, value)

        # Invalidate interpolator cache
        self._interpolator = None

        return MetricResult(
            name=self._name,
            value=value,
            category=self._category,
            direction=self._direction,
            metadata={
                'yaw': yaw,
                'pitch': pitch,
                'yaw_deg': float(np.degrees(yaw)),
                'pitch_deg': float(np.degrees(pitch)),
            },
        )

    def get_spherical_heatmap(
        self,
        resolution: int = 72,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get interpolated score heatmap over the viewing sphere.

        Args:
            resolution: Grid resolution (number of points per dimension)

        Returns:
            Tuple of (yaw_grid, pitch_grid, score_grid) in radians
        """
        if self._interpolator is None:
            scores = self._evaluator.get_all_scores()
            if not scores:
                # Return empty grid
                yaws = np.linspace(0, 2*np.pi, resolution)
                pitches = np.linspace(0, np.pi, resolution)
                yaw_grid, pitch_grid = np.meshgrid(yaws, pitches)
                return yaw_grid, pitch_grid, np.full_like(yaw_grid, 0.5)

            self._interpolator = SphericalScoreInterpolator(
                bandwidth=self.interpolator_bandwidth,
                use_loocv=len(scores) > 10,
            )
            self._interpolator.fit(
                yaws=np.array([s['yaw'] for s in scores]),
                pitches=np.array([s['pitch'] for s in scores]),
                scores=np.array([s['score'] for s in scores]),
            )

        return self._interpolator.predict_grid(resolution=resolution)

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics over all recorded angles."""
        return self._evaluator.compute_statistics()


@MetricRegistry.register("pose_robustness")
class PoseRobustnessMetric(BaseMetric):
    """
    Measures how robust anonymization is across different poses.

    A robust anonymizer should produce consistent identity change
    regardless of viewing angle. High variance indicates pose-dependent
    failures.

    The robustness score is computed as:
        robustness = 1 - (std(scores) / max_possible_std)

    Higher values indicate more consistent performance across poses.

    Example:
        >>> metric = PoseRobustnessMetric()
        >>>
        >>> # Add scores from different viewing angles
        >>> for yaw, pitch, score in angle_scores:
        ...     metric.add_observation(yaw, pitch, score)
        >>>
        >>> result = metric.compute_robustness()
        >>> print(f"Robustness: {result.value:.3f}")
    """

    def __init__(self, **kwargs: Any):
        super().__init__(
            name="pose_robustness",
            category=MetricCategory.EXPLAINABILITY,
            direction=MetricDirection.HIGHER_BETTER,
        )
        self._evaluator = AngleBasedEvaluator("pose_robustness")

    def add_observation(self, yaw: float, pitch: float, score: float) -> None:
        """Add a score observation at a specific pose."""
        self._evaluator.add_score(yaw, pitch, score)

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Not applicable for this metric."""
        return 0.0

    def compute_robustness(self) -> MetricResult:
        """
        Compute overall pose robustness score.

        Returns:
            MetricResult with robustness value and statistics
        """
        import time
        start_time = time.perf_counter()

        stats = self._evaluator.compute_statistics()

        if stats['count'] < 2:
            return MetricResult(
                name=self._name,
                value=1.0,
                category=self._category,
                direction=self._direction,
                metadata={'warning': 'Need at least 2 observations'},
            )

        # Robustness = 1 - normalized_std
        # Assuming scores are in [0, 1], max std is 0.5
        max_std = 0.5
        robustness = 1.0 - min(stats['std'] / max_std, 1.0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Identify weak angles (scores significantly below mean)
        scores = self._evaluator.get_all_scores()
        threshold = stats['mean'] - stats['std']
        weak_angles = [
            {'yaw': s['yaw'], 'pitch': s['pitch'], 'score': s['score']}
            for s in scores if s['score'] < threshold
        ]

        return MetricResult(
            name=self._name,
            value=float(robustness),
            category=self._category,
            direction=self._direction,
            raw_values=np.array([s['score'] for s in scores]),
            metadata={
                **stats,
                'robustness': float(robustness),
                'n_weak_angles': len(weak_angles),
                'weak_angles': weak_angles[:10],  # Top 10 weak angles
            },
            processing_time_ms=elapsed_ms,
        )

    def reset(self) -> None:
        """Clear all observations."""
        self._evaluator = AngleBasedEvaluator("pose_robustness")
