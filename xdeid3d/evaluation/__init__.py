"""
Evaluation framework for X-DeID3D.

This package provides tools for evaluating anonymization quality,
including data structures, sample providers, evaluation modes,
and aggregation utilities.

Sub-modules:
    - data: Core data structures (EvaluationSample, EvaluationResult, etc.)
    - providers: Sample providers for various data sources
    - modes: Evaluation modes (single, aggregate, 3D)
"""

from xdeid3d.evaluation.data import (
    EvaluationSample,
    EvaluationResult,
    AggregatedResults,
    ExperimentMetadata,
    EvaluationStatus,
    SampleType,
)
from xdeid3d.evaluation.providers import (
    SampleProvider,
    DirectorySampleProvider,
    VideoSampleProvider,
    PairedDirectorySampleProvider,
    FrameSequenceProvider,
)
from xdeid3d.evaluation.modes import (
    EvaluationMode,
    MetricSuite,
    SingleSampleMode,
    AggregateMode,
    SphericalMode,
)

__all__ = [
    # Data structures
    "EvaluationSample",
    "EvaluationResult",
    "AggregatedResults",
    "ExperimentMetadata",
    "EvaluationStatus",
    "SampleType",
    # Providers
    "SampleProvider",
    "DirectorySampleProvider",
    "VideoSampleProvider",
    "PairedDirectorySampleProvider",
    "FrameSequenceProvider",
    # Evaluation modes
    "EvaluationMode",
    "MetricSuite",
    "SingleSampleMode",
    "AggregateMode",
    "SphericalMode",
]
