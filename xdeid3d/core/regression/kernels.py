"""
Kernel functions for regression on spherical manifolds.

This module provides kernel functions optimized for use with
spherical distances, particularly for Nadaraya-Watson regression.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np

__all__ = [
    "Kernel",
    "GaussianRBFKernel",
    "EpanechnikovKernel",
    "UniformKernel",
    "VonMisesFisherKernel",
]


class Kernel(ABC):
    """Abstract base class for kernel functions."""

    @abstractmethod
    def __call__(
        self, distance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute kernel weight for given distance(s).

        Args:
            distance: Distance value(s), typically angular distance on S²

        Returns:
            Kernel weight(s), non-negative values
        """
        pass

    @property
    @abstractmethod
    def bandwidth(self) -> float:
        """Return the kernel bandwidth parameter."""
        pass

    @bandwidth.setter
    @abstractmethod
    def bandwidth(self, value: float) -> None:
        """Set the kernel bandwidth parameter."""
        pass


class GaussianRBFKernel(Kernel):
    """
    Gaussian Radial Basis Function kernel.

    K(d) = exp(-d² / (2σ²))

    where d is the distance and σ is the bandwidth parameter.
    This is the most commonly used kernel for spherical regression
    as it produces smooth interpolations and has infinite support.

    Attributes:
        _bandwidth: Standard deviation (σ) of the Gaussian
    """

    def __init__(self, bandwidth: float = 1.0):
        """
        Initialize Gaussian RBF kernel.

        Args:
            bandwidth: Standard deviation σ (controls smoothness)
        """
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")
        self._bandwidth = bandwidth

    def __call__(
        self, distance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute Gaussian kernel weight.

        Args:
            distance: Angular distance(s) in radians

        Returns:
            Kernel weight(s) in range (0, 1]
        """
        # K(d) = exp(-d² / (2σ²))
        return np.exp(-np.square(distance) / (2 * self._bandwidth ** 2))

    @property
    def bandwidth(self) -> float:
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Bandwidth must be positive")
        self._bandwidth = value

    def __repr__(self) -> str:
        return f"GaussianRBFKernel(bandwidth={self._bandwidth:.4f})"


class EpanechnikovKernel(Kernel):
    """
    Epanechnikov kernel with compact support.

    K(d) = (3/4)(1 - (d/h)²) for |d| ≤ h, else 0

    This kernel has optimal efficiency in a minimum variance sense
    and has compact support (zero weight beyond bandwidth).

    Attributes:
        _bandwidth: Support radius h
    """

    def __init__(self, bandwidth: float = 1.0):
        """
        Initialize Epanechnikov kernel.

        Args:
            bandwidth: Support radius h (distances beyond h get zero weight)
        """
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")
        self._bandwidth = bandwidth

    def __call__(
        self, distance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute Epanechnikov kernel weight.

        Args:
            distance: Angular distance(s) in radians

        Returns:
            Kernel weight(s), zero for distances > bandwidth
        """
        u = distance / self._bandwidth
        # K(u) = (3/4)(1 - u²) for |u| ≤ 1
        weights = 0.75 * (1 - np.square(u))
        # Set to zero outside support
        if isinstance(weights, np.ndarray):
            weights = np.where(np.abs(u) <= 1, weights, 0.0)
        else:
            weights = weights if abs(u) <= 1 else 0.0
        return np.maximum(weights, 0.0)

    @property
    def bandwidth(self) -> float:
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Bandwidth must be positive")
        self._bandwidth = value

    def __repr__(self) -> str:
        return f"EpanechnikovKernel(bandwidth={self._bandwidth:.4f})"


class UniformKernel(Kernel):
    """
    Uniform (box) kernel with compact support.

    K(d) = 1/2 for |d| ≤ h, else 0

    This kernel gives equal weight to all points within the bandwidth
    and zero weight to points outside.

    Attributes:
        _bandwidth: Support radius h
    """

    def __init__(self, bandwidth: float = 1.0):
        """
        Initialize Uniform kernel.

        Args:
            bandwidth: Support radius h
        """
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")
        self._bandwidth = bandwidth

    def __call__(
        self, distance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute Uniform kernel weight.

        Args:
            distance: Angular distance(s) in radians

        Returns:
            Kernel weight(s), 0.5 within bandwidth, 0 outside
        """
        if isinstance(distance, np.ndarray):
            return np.where(distance <= self._bandwidth, 0.5, 0.0)
        return 0.5 if distance <= self._bandwidth else 0.0

    @property
    def bandwidth(self) -> float:
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Bandwidth must be positive")
        self._bandwidth = value

    def __repr__(self) -> str:
        return f"UniformKernel(bandwidth={self._bandwidth:.4f})"


class VonMisesFisherKernel(Kernel):
    """
    Von Mises-Fisher kernel for directional data.

    K(d) = exp(κ * cos(d))

    This kernel is the natural choice for directional data on S²
    as it is the spherical analogue of the normal distribution.

    The concentration parameter κ controls the bandwidth:
    - κ → 0: uniform distribution on sphere
    - κ → ∞: concentrated around mean direction

    Attributes:
        _kappa: Concentration parameter
    """

    def __init__(self, kappa: float = 1.0):
        """
        Initialize Von Mises-Fisher kernel.

        Args:
            kappa: Concentration parameter (κ ≥ 0)
        """
        if kappa < 0:
            raise ValueError("Concentration parameter must be non-negative")
        self._kappa = kappa

    def __call__(
        self, distance: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute Von Mises-Fisher kernel weight.

        Args:
            distance: Angular distance(s) in radians

        Returns:
            Kernel weight(s)
        """
        # K(d) = exp(κ * cos(d))
        # Note: cos(0) = 1, cos(π) = -1
        return np.exp(self._kappa * np.cos(distance))

    @property
    def bandwidth(self) -> float:
        """Return equivalent bandwidth (inverse of kappa)."""
        return 1.0 / max(self._kappa, 1e-8)

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        """Set bandwidth (as inverse of kappa)."""
        if value <= 0:
            raise ValueError("Bandwidth must be positive")
        self._kappa = 1.0 / value

    @property
    def kappa(self) -> float:
        """Return concentration parameter."""
        return self._kappa

    @kappa.setter
    def kappa(self, value: float) -> None:
        """Set concentration parameter."""
        if value < 0:
            raise ValueError("Concentration parameter must be non-negative")
        self._kappa = value

    def __repr__(self) -> str:
        return f"VonMisesFisherKernel(kappa={self._kappa:.4f})"
