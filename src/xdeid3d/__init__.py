"""
X-DeID3D: Explainable 3D Auditing Framework for Face De-identification Systems.

This package provides tools for evaluating face anonymization systems using
continuous 3D explanations based on kernel regression on spherical manifolds.
"""

__version__ = "0.1.0"
__author__ = "X-DeID3D Contributors"

# Lazy imports to avoid circular dependencies and heavy imports at startup
def __getattr__(name: str):
    """Lazy import module attributes."""
    if name == "geometry":
        from xdeid3d.core import geometry
        return geometry
    elif name == "regression":
        from xdeid3d.core import regression
        return regression
    elif name == "BaseAnonymizer":
        from xdeid3d.anonymizers.base import BaseAnonymizer
        return BaseAnonymizer
    elif name == "AnonymizerRegistry":
        from xdeid3d.anonymizers.registry import AnonymizerRegistry
        return AnonymizerRegistry
    elif name == "BaseMetric":
        from xdeid3d.metrics.base import BaseMetric
        return BaseMetric
    elif name == "MetricRegistry":
        from xdeid3d.metrics.registry import MetricRegistry
        return MetricRegistry
    elif name == "EvaluationPipeline":
        from xdeid3d.evaluation.pipeline import EvaluationPipeline
        return EvaluationPipeline
    elif name == "Config":
        from xdeid3d.config.schema import Config
        return Config
    elif name == "synthesis":
        from xdeid3d import synthesis
        return synthesis
    elif name == "CameraPoseSampler":
        from xdeid3d.synthesis.camera import CameraPoseSampler
        return CameraPoseSampler
    elif name == "BasicRenderer":
        from xdeid3d.synthesis.rendering import BasicRenderer
        return BasicRenderer
    elif name == "spherehead":
        from xdeid3d.spherehead import __init__ as _spherehead_init
        import xdeid3d.spherehead
        return xdeid3d.spherehead
    elif name == "TriPlaneGenerator":
        from xdeid3d.spherehead import TriPlaneGenerator
        return TriPlaneGenerator
    elif name == "load_network":
        from xdeid3d.spherehead import load_network
        return load_network
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "geometry",
    "regression",
    "BaseAnonymizer",
    "AnonymizerRegistry",
    "BaseMetric",
    "MetricRegistry",
    "EvaluationPipeline",
    "Config",
    "synthesis",
    "CameraPoseSampler",
    "BasicRenderer",
    # SphereHead synthesis engine
    "spherehead",
    "TriPlaneGenerator",
    "load_network",
]