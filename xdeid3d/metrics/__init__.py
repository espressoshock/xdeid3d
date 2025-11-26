"""
Metrics framework for X-DeID3D.

This package provides metrics for evaluating face anonymization quality,
including identity metrics, visual quality metrics, temporal consistency,
and explainability metrics for 3D analysis.

Sub-packages:
    - identity: ArcFace-based identity change metrics
    - quality: PSNR, SSIM, LPIPS visual quality metrics
    - temporal: TIC, TVS temporal consistency metrics
    - explainability: Viewpoint, saliency, and composite analysis
"""

from xdeid3d.metrics.base import (
    MetricProtocol,
    BaseMetric,
    MetricResult,
    MetricCategory,
    MetricDirection,
    AggregationMethod,
)
from xdeid3d.metrics.registry import (
    MetricRegistry,
    register_metric,
    create_metric,
)

__all__ = [
    # Base classes
    "MetricProtocol",
    "BaseMetric",
    "MetricResult",
    "MetricCategory",
    "MetricDirection",
    "AggregationMethod",
    # Registry
    "MetricRegistry",
    "register_metric",
    "create_metric",
]