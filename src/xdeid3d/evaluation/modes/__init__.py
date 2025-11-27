"""
Evaluation modes for X-DeID3D.

This package provides different evaluation modes:
    - SingleSampleMode: Evaluate individual samples
    - AggregateMode: Population-level statistics
    - SphericalMode: 3D spherical evaluation with pose tracking
"""

from xdeid3d.evaluation.modes.base import (
    EvaluationMode,
    MetricSuite,
)
from xdeid3d.evaluation.modes.single import SingleSampleMode
from xdeid3d.evaluation.modes.aggregate import AggregateMode
from xdeid3d.evaluation.modes.spherical import SphericalMode

__all__ = [
    # Base classes
    "EvaluationMode",
    "MetricSuite",
    # Evaluation modes
    "SingleSampleMode",
    "AggregateMode",
    "SphericalMode",
]
