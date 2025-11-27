"""
Kernel regression on spherical manifolds.

This module implements Nadaraya-Watson kernel regression for
interpolating sparse measurements on S² to dense heatmaps.
"""

from xdeid3d.core.regression.kernels import (
    Kernel,
    GaussianRBFKernel,
    EpanechnikovKernel,
    UniformKernel,
    VonMisesFisherKernel,
)
from xdeid3d.core.regression.nadaraya_watson import (
    NadarayaWatsonEstimator,
    SphericalRegressionResult,
)
from xdeid3d.core.regression.bandwidth import (
    LOOCVBandwidthSelector,
    RuleOfThumbBandwidth,
    GridSearchBandwidth,
)

__all__ = [
    # Kernels
    "Kernel",
    "GaussianRBFKernel",
    "EpanechnikovKernel",
    "UniformKernel",
    "VonMisesFisherKernel",
    # Estimator
    "NadarayaWatsonEstimator",
    "SphericalRegressionResult",
    # Bandwidth selection
    "LOOCVBandwidthSelector",
    "RuleOfThumbBandwidth",
    "GridSearchBandwidth",
]