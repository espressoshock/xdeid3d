"""
Metrics framework for X-DeID3D.

This package provides metrics for evaluating face anonymization quality,
including identity metrics, visual quality metrics, and temporal consistency.
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