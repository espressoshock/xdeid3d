"""
Composite explainability metrics for comprehensive anonymization analysis.

These metrics combine multiple aspects of explainability into unified
scores and visualizations.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from xdeid3d.metrics.base import BaseMetric, MetricCategory, MetricDirection, MetricResult
from xdeid3d.metrics.registry import MetricRegistry

__all__ = [
    "ComprehensiveExplainabilityMetric",
    "FailureModeDetector",
    "AnonymizationQualityIndex",
]


@MetricRegistry.register("comprehensive_explainability")
class ComprehensiveExplainabilityMetric(BaseMetric):
    """
    Comprehensive explainability metric combining multiple analysis types.

    Aggregates viewpoint robustness, regional saliency, and identity
    contribution into a single explainability score with detailed breakdown.

    Args:
        weights: Dictionary of weights for each sub-metric

    Example:
        >>> metric = ComprehensiveExplainabilityMetric()
        >>> result = metric.compute_comprehensive(
        ...     original_frames, anonymized_frames, poses
        ... )
        >>> print(result.metadata['breakdown'])
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="comprehensive_explainability",
            category=MetricCategory.EXPLAINABILITY,
            direction=MetricDirection.HIGHER_BETTER,
        )
        self.weights = weights or {
            'pose_robustness': 0.4,
            'regional_consistency': 0.3,
            'identity_change_magnitude': 0.3,
        }

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute single-frame explainability score."""
        from xdeid3d.metrics.explainability.saliency import RegionalSaliencyMetric

        # Regional saliency as proxy for single-frame
        saliency = RegionalSaliencyMetric()
        result = saliency.compute(original, anonymized)

        return result.value

    def compute_comprehensive(
        self,
        original_frames: Sequence[np.ndarray],
        anonymized_frames: Sequence[np.ndarray],
        poses: Optional[Sequence[tuple]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute comprehensive explainability over a sequence.

        Args:
            original_frames: Sequence of original frames
            anonymized_frames: Sequence of anonymized frames
            poses: Optional sequence of (yaw, pitch) tuples

        Returns:
            MetricResult with comprehensive analysis
        """
        import time
        from xdeid3d.metrics.explainability.viewpoint import PoseRobustnessMetric
        from xdeid3d.metrics.explainability.saliency import RegionalSaliencyMetric
        from xdeid3d.metrics.identity.embeddings import InsightFaceExtractor, cosine_distance

        start_time = time.perf_counter()
        self._ensure_initialized()

        n_frames = len(original_frames)
        breakdown = {}

        # 1. Pose robustness (if poses provided)
        pose_robustness_score = 1.0
        if poses is not None and len(poses) == n_frames:
            pose_metric = PoseRobustnessMetric()

            # Compute identity distance for each frame
            try:
                extractor = InsightFaceExtractor()
                for i, ((orig, anon), (yaw, pitch)) in enumerate(
                    zip(zip(original_frames, anonymized_frames), poses)
                ):
                    orig_emb = extractor.extract(orig)
                    anon_emb = extractor.extract(anon)
                    if orig_emb is not None and anon_emb is not None:
                        dist = cosine_distance(orig_emb, anon_emb)
                        pose_metric.add_observation(yaw, pitch, dist)

                robustness_result = pose_metric.compute_robustness()
                pose_robustness_score = robustness_result.value
                breakdown['pose_robustness'] = {
                    'score': pose_robustness_score,
                    'stats': robustness_result.metadata,
                }
            except Exception as e:
                breakdown['pose_robustness'] = {
                    'score': 1.0,
                    'error': str(e),
                }

        # 2. Regional consistency across frames
        regional_scores = []
        saliency_metric = RegionalSaliencyMetric(grid_size=8)

        for orig, anon in zip(original_frames, anonymized_frames):
            result = saliency_metric.compute(orig, anon)
            regional_scores.append(result.value)

        regional_consistency = 1.0 - np.std(regional_scores)  # Higher = more consistent
        breakdown['regional_consistency'] = {
            'score': float(regional_consistency),
            'mean_saliency': float(np.mean(regional_scores)),
            'std_saliency': float(np.std(regional_scores)),
        }

        # 3. Identity change magnitude
        identity_scores = []
        try:
            extractor = InsightFaceExtractor()
            for orig, anon in zip(original_frames, anonymized_frames):
                orig_emb = extractor.extract(orig)
                anon_emb = extractor.extract(anon)
                if orig_emb is not None and anon_emb is not None:
                    dist = cosine_distance(orig_emb, anon_emb)
                    identity_scores.append(dist)

            if identity_scores:
                identity_change = float(np.mean(identity_scores))
                breakdown['identity_change'] = {
                    'score': identity_change,
                    'std': float(np.std(identity_scores)),
                    'min': float(np.min(identity_scores)),
                    'max': float(np.max(identity_scores)),
                }
            else:
                identity_change = 0.5
                breakdown['identity_change'] = {'score': 0.5, 'note': 'No faces detected'}
        except Exception as e:
            identity_change = 0.5
            breakdown['identity_change'] = {'score': 0.5, 'error': str(e)}

        # Compute weighted composite score
        composite_score = (
            self.weights.get('pose_robustness', 0.4) * pose_robustness_score +
            self.weights.get('regional_consistency', 0.3) * regional_consistency +
            self.weights.get('identity_change_magnitude', 0.3) * identity_change
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MetricResult(
            name=self._name,
            value=float(composite_score),
            category=self._category,
            direction=self._direction,
            metadata={
                'breakdown': breakdown,
                'weights': self.weights,
                'n_frames': n_frames,
            },
            processing_time_ms=elapsed_ms,
        )


@MetricRegistry.register("failure_mode_detector")
class FailureModeDetector(BaseMetric):
    """
    Detects and categorizes failure modes in anonymization.

    Identifies specific types of failures:
    - Identity leakage (anonymized too similar to original)
    - Excessive distortion (quality degradation)
    - Temporal inconsistency (flickering, identity drift)
    - Pose-dependent failures (works only at certain angles)

    Example:
        >>> detector = FailureModeDetector()
        >>> result = detector.detect_failures(
        ...     original_frames, anonymized_frames, poses
        ... )
        >>> failures = result.metadata['detected_failures']
    """

    def __init__(
        self,
        identity_threshold: float = 0.3,
        quality_threshold: float = 0.7,
        consistency_threshold: float = 0.9,
        **kwargs: Any,
    ):
        super().__init__(
            name="failure_mode_detector",
            category=MetricCategory.EXPLAINABILITY,
            direction=MetricDirection.LOWER_BETTER,  # Lower = fewer failures
        )
        self.identity_threshold = identity_threshold
        self.quality_threshold = quality_threshold
        self.consistency_threshold = consistency_threshold

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Return number of failure types detected."""
        return 0.0  # Use detect_failures for full analysis

    def detect_failures(
        self,
        original_frames: Sequence[np.ndarray],
        anonymized_frames: Sequence[np.ndarray],
        poses: Optional[Sequence[tuple]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Detect failure modes across a sequence.

        Returns:
            MetricResult with detected failures categorized
        """
        import time
        start_time = time.perf_counter()

        failures: List[Dict[str, Any]] = []
        n_frames = len(original_frames)

        # Check identity leakage
        identity_failures = self._check_identity_leakage(
            original_frames, anonymized_frames
        )
        failures.extend(identity_failures)

        # Check quality degradation
        quality_failures = self._check_quality_degradation(
            original_frames, anonymized_frames
        )
        failures.extend(quality_failures)

        # Check temporal consistency (if multiple frames)
        if n_frames > 1:
            temporal_failures = self._check_temporal_consistency(anonymized_frames)
            failures.extend(temporal_failures)

        # Check pose-dependent failures (if poses provided)
        if poses is not None:
            pose_failures = self._check_pose_dependent_failures(
                original_frames, anonymized_frames, poses
            )
            failures.extend(pose_failures)

        # Categorize failures
        failure_counts = {}
        for f in failures:
            ftype = f.get('type', 'unknown')
            failure_counts[ftype] = failure_counts.get(ftype, 0) + 1

        # Overall failure score (0 = no failures, 1 = many failures)
        total_possible_failures = n_frames * 4  # 4 failure types per frame
        failure_score = min(len(failures) / max(total_possible_failures, 1), 1.0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MetricResult(
            name=self._name,
            value=float(failure_score),
            category=self._category,
            direction=self._direction,
            metadata={
                'detected_failures': failures,
                'failure_counts': failure_counts,
                'total_failures': len(failures),
                'n_frames': n_frames,
                'thresholds': {
                    'identity': self.identity_threshold,
                    'quality': self.quality_threshold,
                    'consistency': self.consistency_threshold,
                },
            },
            processing_time_ms=elapsed_ms,
        )

    def _check_identity_leakage(
        self,
        original_frames: Sequence[np.ndarray],
        anonymized_frames: Sequence[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Check for identity leakage (anonymized too similar to original)."""
        failures = []

        try:
            from xdeid3d.metrics.identity.embeddings import (
                InsightFaceExtractor, cosine_distance
            )
            extractor = InsightFaceExtractor()

            for i, (orig, anon) in enumerate(zip(original_frames, anonymized_frames)):
                orig_emb = extractor.extract(orig)
                anon_emb = extractor.extract(anon)

                if orig_emb is not None and anon_emb is not None:
                    dist = cosine_distance(orig_emb, anon_emb)

                    if dist < self.identity_threshold:
                        failures.append({
                            'type': 'identity_leakage',
                            'frame': i,
                            'severity': 1.0 - dist / self.identity_threshold,
                            'value': dist,
                            'description': f'Identity distance {dist:.3f} below threshold {self.identity_threshold}',
                        })
        except ImportError:
            pass

        return failures

    def _check_quality_degradation(
        self,
        original_frames: Sequence[np.ndarray],
        anonymized_frames: Sequence[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Check for excessive quality degradation."""
        failures = []

        try:
            from xdeid3d.metrics.quality.ssim import SSIMMetric
            ssim_metric = SSIMMetric()

            for i, (orig, anon) in enumerate(zip(original_frames, anonymized_frames)):
                result = ssim_metric.compute(orig, anon)
                ssim_value = result.value

                if ssim_value < self.quality_threshold:
                    failures.append({
                        'type': 'quality_degradation',
                        'frame': i,
                        'severity': 1.0 - ssim_value / self.quality_threshold,
                        'value': ssim_value,
                        'description': f'SSIM {ssim_value:.3f} below threshold {self.quality_threshold}',
                    })
        except ImportError:
            pass

        return failures

    def _check_temporal_consistency(
        self,
        anonymized_frames: Sequence[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Check for temporal inconsistency (flickering)."""
        failures = []

        try:
            from xdeid3d.metrics.identity.embeddings import (
                InsightFaceExtractor, cosine_distance
            )
            extractor = InsightFaceExtractor()

            embeddings = []
            for frame in anonymized_frames:
                emb = extractor.extract(frame)
                embeddings.append(emb)

            for i in range(len(embeddings) - 1):
                if embeddings[i] is not None and embeddings[i+1] is not None:
                    dist = cosine_distance(embeddings[i], embeddings[i+1])

                    if dist > (1.0 - self.consistency_threshold):
                        failures.append({
                            'type': 'temporal_inconsistency',
                            'frame': i,
                            'severity': dist,
                            'value': dist,
                            'description': f'Identity change {dist:.3f} between frames {i} and {i+1}',
                        })
        except ImportError:
            pass

        return failures

    def _check_pose_dependent_failures(
        self,
        original_frames: Sequence[np.ndarray],
        anonymized_frames: Sequence[np.ndarray],
        poses: Sequence[tuple],
    ) -> List[Dict[str, Any]]:
        """Check for pose-dependent failures."""
        failures = []

        try:
            from xdeid3d.metrics.identity.embeddings import (
                InsightFaceExtractor, cosine_distance
            )
            extractor = InsightFaceExtractor()

            # Compute identity distances per pose
            distances = []
            for i, ((orig, anon), (yaw, pitch)) in enumerate(
                zip(zip(original_frames, anonymized_frames), poses)
            ):
                orig_emb = extractor.extract(orig)
                anon_emb = extractor.extract(anon)

                if orig_emb is not None and anon_emb is not None:
                    dist = cosine_distance(orig_emb, anon_emb)
                    distances.append({
                        'frame': i,
                        'yaw': yaw,
                        'pitch': pitch,
                        'distance': dist,
                    })

            if len(distances) > 3:
                mean_dist = np.mean([d['distance'] for d in distances])
                std_dist = np.std([d['distance'] for d in distances])

                # Flag frames with unusually low identity distance
                for d in distances:
                    if d['distance'] < mean_dist - 2 * std_dist:
                        failures.append({
                            'type': 'pose_dependent_failure',
                            'frame': d['frame'],
                            'severity': (mean_dist - d['distance']) / (2 * std_dist + 1e-8),
                            'value': d['distance'],
                            'yaw': d['yaw'],
                            'pitch': d['pitch'],
                            'description': f'Low identity distance {d["distance"]:.3f} at yaw={np.degrees(d["yaw"]):.1f}°',
                        })
        except ImportError:
            pass

        return failures


@MetricRegistry.register("anonymization_quality_index")
class AnonymizationQualityIndex(BaseMetric):
    """
    Single-number quality index for anonymization.

    Combines identity protection, visual quality, and temporal
    consistency into a normalized score from 0-100.

    Formula:
        AQI = w_id * ID_score + w_quality * Quality_score + w_temporal * Temporal_score

    Where:
        - ID_score: Identity change (higher = better protection)
        - Quality_score: SSIM-based visual quality
        - Temporal_score: Temporal consistency (TIC)

    Example:
        >>> metric = AnonymizationQualityIndex()
        >>> result = metric.compute_index(original_frames, anonymized_frames)
        >>> print(f"AQI: {result.value:.1f}/100")
    """

    def __init__(
        self,
        identity_weight: float = 0.5,
        quality_weight: float = 0.3,
        temporal_weight: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__(
            name="anonymization_quality_index",
            category=MetricCategory.EXPLAINABILITY,
            direction=MetricDirection.HIGHER_BETTER,
        )
        self.identity_weight = identity_weight
        self.quality_weight = quality_weight
        self.temporal_weight = temporal_weight

    def _compute_single(
        self,
        original: np.ndarray,
        anonymized: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """Compute AQI for single frame."""
        # Single frame version (no temporal component)
        scores = {}

        # Identity score
        try:
            from xdeid3d.metrics.identity.embeddings import (
                InsightFaceExtractor, cosine_distance
            )
            extractor = InsightFaceExtractor()
            orig_emb = extractor.extract(original)
            anon_emb = extractor.extract(anonymized)

            if orig_emb is not None and anon_emb is not None:
                scores['identity'] = cosine_distance(orig_emb, anon_emb)
            else:
                scores['identity'] = 0.5
        except ImportError:
            scores['identity'] = 0.5

        # Quality score
        try:
            from xdeid3d.metrics.quality.ssim import SSIMMetric
            ssim = SSIMMetric()
            scores['quality'] = ssim.compute(original, anonymized).value
        except ImportError:
            scores['quality'] = 0.5

        # Compute weighted average (no temporal for single frame)
        total_weight = self.identity_weight + self.quality_weight
        aqi = (
            self.identity_weight * scores['identity'] +
            self.quality_weight * scores['quality']
        ) / total_weight

        return float(aqi * 100)

    def compute_index(
        self,
        original_frames: Sequence[np.ndarray],
        anonymized_frames: Sequence[np.ndarray],
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute AQI over a sequence.

        Returns:
            MetricResult with AQI score (0-100) and component breakdown
        """
        import time
        start_time = time.perf_counter()

        scores = {
            'identity': [],
            'quality': [],
        }

        # Compute per-frame scores
        try:
            from xdeid3d.metrics.identity.embeddings import (
                InsightFaceExtractor, cosine_distance
            )
            extractor = InsightFaceExtractor()

            for orig, anon in zip(original_frames, anonymized_frames):
                orig_emb = extractor.extract(orig)
                anon_emb = extractor.extract(anon)

                if orig_emb is not None and anon_emb is not None:
                    scores['identity'].append(cosine_distance(orig_emb, anon_emb))
        except ImportError:
            pass

        try:
            from xdeid3d.metrics.quality.ssim import SSIMMetric
            ssim = SSIMMetric()

            for orig, anon in zip(original_frames, anonymized_frames):
                result = ssim.compute(orig, anon)
                scores['quality'].append(result.value)
        except ImportError:
            pass

        # Compute temporal score (TIC)
        temporal_score = 1.0
        try:
            from xdeid3d.metrics.temporal.consistency import TemporalIdentityConsistency
            tic = TemporalIdentityConsistency()
            tic_result = tic.compute_sequence(anonymized_frames)
            temporal_score = tic_result.value
        except ImportError:
            pass

        # Aggregate scores
        identity_mean = float(np.mean(scores['identity'])) if scores['identity'] else 0.5
        quality_mean = float(np.mean(scores['quality'])) if scores['quality'] else 0.5

        # Compute weighted AQI
        aqi = (
            self.identity_weight * identity_mean +
            self.quality_weight * quality_mean +
            self.temporal_weight * temporal_score
        )
        aqi_100 = aqi * 100

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MetricResult(
            name=self._name,
            value=float(aqi_100),
            category=self._category,
            direction=self._direction,
            metadata={
                'components': {
                    'identity': {
                        'score': identity_mean,
                        'weight': self.identity_weight,
                        'weighted': identity_mean * self.identity_weight,
                    },
                    'quality': {
                        'score': quality_mean,
                        'weight': self.quality_weight,
                        'weighted': quality_mean * self.quality_weight,
                    },
                    'temporal': {
                        'score': temporal_score,
                        'weight': self.temporal_weight,
                        'weighted': temporal_score * self.temporal_weight,
                    },
                },
                'aqi_raw': float(aqi),
                'n_frames': len(original_frames),
            },
            processing_time_ms=elapsed_ms,
        )
