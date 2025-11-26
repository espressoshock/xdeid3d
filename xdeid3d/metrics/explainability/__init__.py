"""
Explainability metrics for anonymization evaluation.

This module provides metrics for understanding and explaining
anonymization behavior across different viewing angles, regions,
and conditions.

Key Components:
    - ViewpointMetric: Tracks metrics across viewing angles
    - SphericalScoreInterpolator: Interpolates scores on S² using kernel regression
    - RegionalSaliencyMetric: Identifies identity-relevant regions
    - FailureModeDetector: Detects and categorizes failure types
    - AnonymizationQualityIndex: Single-number quality score (0-100)
"""

from xdeid3d.metrics.explainability.viewpoint import (
    ViewpointMetric,
    SphericalScoreInterpolator,
    PoseRobustnessMetric,
    AngleBasedEvaluator,
)
from xdeid3d.metrics.explainability.saliency import (
    RegionalSaliencyMetric,
    DifferenceMapMetric,
    IdentityContributionMap,
)
from xdeid3d.metrics.explainability.composite import (
    ComprehensiveExplainabilityMetric,
    FailureModeDetector,
    AnonymizationQualityIndex,
)

__all__ = [
    # Viewpoint-based explainability
    "ViewpointMetric",
    "SphericalScoreInterpolator",
    "PoseRobustnessMetric",
    "AngleBasedEvaluator",
    # Regional saliency
    "RegionalSaliencyMetric",
    "DifferenceMapMetric",
    "IdentityContributionMap",
    # Composite metrics
    "ComprehensiveExplainabilityMetric",
    "FailureModeDetector",
    "AnonymizationQualityIndex",
]
