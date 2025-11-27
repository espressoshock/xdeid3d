"""
Nadaraya-Watson kernel regression estimator.

This module implements the Nadaraya-Watson estimator for non-parametric
regression, specialized for use on spherical manifolds (S²).

The estimator computes:
    ŷ(x) = Σᵢ K(d(x, xᵢ)) · yᵢ / Σᵢ K(d(x, xᵢ))

where:
    - K is a kernel function
    - d(·, ·) is a distance function (great-circle distance for S²)
    - (xᵢ, yᵢ) are training samples
"""

from typing import List, Optional, Tuple, Union, Callable
import numpy as np

from .kernels import Kernel, GaussianRBFKernel
from ..geometry.spherical import great_circle_distance

__all__ = [
    "NadarayaWatsonEstimator",
    "SphericalRegressionResult",
]


class SphericalRegressionResult:
    """
    Result of spherical regression at a query point.

    Attributes:
        value: Estimated value at the query point
        weight_sum: Total weight from all training points
        distances: Distances to all training points
        weights: Kernel weights for all training points
    """

    __slots__ = ("value", "weight_sum", "distances", "weights")

    def __init__(
        self,
        value: float,
        weight_sum: float,
        distances: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        self.value = value
        self.weight_sum = weight_sum
        self.distances = distances
        self.weights = weights

    @property
    def is_extrapolation(self) -> bool:
        """
        Check if this is an extrapolation (low total weight).

        Returns True if the query point is far from all training points.
        """
        return self.weight_sum < 0.01

    @property
    def effective_samples(self) -> float:
        """
        Estimate effective number of samples contributing.

        Uses the formula: n_eff = (Σwᵢ)² / Σwᵢ²
        """
        if self.weights is None or len(self.weights) == 0:
            return 0.0
        return self.weight_sum ** 2 / np.sum(self.weights ** 2)


class NadarayaWatsonEstimator:
    """
    Nadaraya-Watson kernel regression estimator for spherical data.

    This estimator performs non-parametric regression using kernel
    smoothing, computing a weighted average of training values where
    weights are determined by kernel-evaluated distances.

    The estimator supports:
    - Customizable kernel functions
    - Great-circle distance on S²
    - Batch prediction for multiple query points
    - Leave-one-out cross-validation for bandwidth selection

    Example:
        >>> estimator = NadarayaWatsonEstimator(bandwidth=0.5)
        >>> estimator.fit(angles, values)
        >>> prediction = estimator.predict(query_angle)
    """

    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        bandwidth: float = 0.5,
    ):
        """
        Initialize Nadaraya-Watson estimator.

        Args:
            kernel: Kernel function (default: GaussianRBFKernel)
            bandwidth: Kernel bandwidth parameter
        """
        if kernel is None:
            kernel = GaussianRBFKernel(bandwidth=bandwidth)
        else:
            kernel.bandwidth = bandwidth

        self.kernel = kernel
        self._train_angles: Optional[np.ndarray] = None
        self._train_values: Optional[np.ndarray] = None

    def fit(
        self,
        angles: np.ndarray,
        values: np.ndarray,
    ) -> "NadarayaWatsonEstimator":
        """
        Fit the estimator with training data.

        Args:
            angles: Training angles as (N, 2) array of (theta, phi) pairs
                    or list of dicts with 'yaw' and 'pitch' keys
            values: Training values as (N,) array

        Returns:
            self for method chaining
        """
        # Handle dict input
        if isinstance(angles, list) and len(angles) > 0 and isinstance(angles[0], dict):
            angles = np.array([[a['yaw'], a['pitch']] for a in angles])

        angles = np.asarray(angles)
        values = np.asarray(values)

        if angles.ndim == 1:
            raise ValueError("angles must be 2D array with shape (N, 2)")

        if len(angles) != len(values):
            raise ValueError("angles and values must have same length")

        self._train_angles = angles
        self._train_values = values

        return self

    def predict(
        self,
        query_angle: Union[np.ndarray, Tuple[float, float]],
        return_details: bool = False,
    ) -> Union[float, SphericalRegressionResult]:
        """
        Predict value at a query angle.

        Args:
            query_angle: Query point as (theta, phi) tuple or 1D array
            return_details: If True, return SphericalRegressionResult

        Returns:
            Predicted value, or SphericalRegressionResult if return_details=True
        """
        if self._train_angles is None:
            raise RuntimeError("Estimator must be fit before prediction")

        query = np.asarray(query_angle)
        if query.ndim == 0 or len(query) != 2:
            raise ValueError("query_angle must have 2 elements (theta, phi)")

        # Compute distances to all training points
        distances = great_circle_distance(
            query[0], query[1],
            self._train_angles[:, 0], self._train_angles[:, 1]
        )

        # Compute kernel weights
        weights = self.kernel(distances)

        # Nadaraya-Watson estimator: weighted average
        weight_sum = np.sum(weights)

        if weight_sum < 1e-10:
            # All points have near-zero weight, use mean as fallback
            value = np.mean(self._train_values)
        else:
            value = np.sum(weights * self._train_values) / weight_sum

        if return_details:
            return SphericalRegressionResult(
                value=value,
                weight_sum=weight_sum,
                distances=distances,
                weights=weights,
            )

        return value

    def predict_batch(
        self,
        query_angles: np.ndarray,
    ) -> np.ndarray:
        """
        Predict values at multiple query angles.

        Args:
            query_angles: Query points as (M, 2) array of (theta, phi) pairs

        Returns:
            Predicted values as (M,) array
        """
        query_angles = np.asarray(query_angles)
        if query_angles.ndim != 2 or query_angles.shape[1] != 2:
            raise ValueError("query_angles must have shape (M, 2)")

        predictions = np.zeros(len(query_angles))
        for i, query in enumerate(query_angles):
            predictions[i] = self.predict(query)

        return predictions

    def predict_grid(
        self,
        theta_range: Tuple[float, float] = (0, 2 * np.pi),
        phi_range: Tuple[float, float] = (0, np.pi),
        resolution: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict values on a regular grid over S².

        Args:
            theta_range: Range of theta (azimuthal angle)
            phi_range: Range of phi (polar angle)
            resolution: Number of points per dimension

        Returns:
            theta_grid: 2D array of theta values
            phi_grid: 2D array of phi values
            value_grid: 2D array of predicted values
        """
        theta = np.linspace(theta_range[0], theta_range[1], resolution)
        phi = np.linspace(phi_range[0], phi_range[1], resolution)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        # Flatten for batch prediction
        query_angles = np.column_stack([
            theta_grid.ravel(),
            phi_grid.ravel()
        ])

        values = self.predict_batch(query_angles)
        value_grid = values.reshape(theta_grid.shape)

        return theta_grid, phi_grid, value_grid

    def leave_one_out_error(self) -> float:
        """
        Compute leave-one-out cross-validation error.

        Returns:
            Mean squared error from LOO-CV
        """
        if self._train_angles is None:
            raise RuntimeError("Estimator must be fit before LOO-CV")

        n = len(self._train_values)
        errors = np.zeros(n)

        for i in range(n):
            # Compute distances to all other points
            distances = great_circle_distance(
                self._train_angles[i, 0], self._train_angles[i, 1],
                self._train_angles[:, 0], self._train_angles[:, 1]
            )

            # Compute weights, excluding self (distance = 0)
            weights = self.kernel(distances)
            weights[i] = 0  # Leave out point i

            weight_sum = np.sum(weights)

            if weight_sum < 1e-10:
                # Fallback to mean of other points
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                prediction = np.mean(self._train_values[mask])
            else:
                prediction = np.sum(weights * self._train_values) / weight_sum

            errors[i] = (prediction - self._train_values[i]) ** 2

        return np.mean(errors)

    @property
    def bandwidth(self) -> float:
        """Get current bandwidth."""
        return self.kernel.bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        """Set bandwidth."""
        self.kernel.bandwidth = value

    @property
    def n_samples(self) -> int:
        """Number of training samples."""
        return len(self._train_values) if self._train_values is not None else 0

    def __repr__(self) -> str:
        n = self.n_samples
        return f"NadarayaWatsonEstimator(kernel={self.kernel}, n_samples={n})"
