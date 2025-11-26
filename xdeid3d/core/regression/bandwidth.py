"""
Bandwidth selection methods for kernel regression.

This module provides methods for automatically selecting the optimal
bandwidth parameter for Nadaraya-Watson kernel regression, including
Leave-One-Out Cross-Validation (LOOCV).
"""

from typing import List, Optional, Tuple, Union, Callable
import numpy as np
from scipy.optimize import minimize_scalar, brent

from .kernels import Kernel, GaussianRBFKernel
from .nadaraya_watson import NadarayaWatsonEstimator
from ..geometry.spherical import great_circle_distance

__all__ = [
    "LOOCVBandwidthSelector",
    "RuleOfThumbBandwidth",
    "GridSearchBandwidth",
]


class LOOCVBandwidthSelector:
    """
    Leave-One-Out Cross-Validation bandwidth selector.

    This selector finds the bandwidth that minimizes the LOO-CV error,
    which estimates the prediction error on unseen data.

    The algorithm:
    1. For each candidate bandwidth h
    2. For each training point i:
       - Predict yᵢ using all other points
       - Compute squared error (ŷᵢ - yᵢ)²
    3. Return bandwidth with minimum mean squared error

    Example:
        >>> selector = LOOCVBandwidthSelector()
        >>> optimal_h = selector.select(angles, values)
        >>> estimator = NadarayaWatsonEstimator(bandwidth=optimal_h)
    """

    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        bandwidth_range: Tuple[float, float] = (0.01, 2.0),
        n_candidates: int = 50,
        optimizer: str = "brent",
    ):
        """
        Initialize LOOCV bandwidth selector.

        Args:
            kernel: Kernel to use (default: GaussianRBFKernel)
            bandwidth_range: Search range for bandwidth (min, max)
            n_candidates: Number of candidates for grid search
            optimizer: Optimization method ("brent" or "grid")
        """
        self.kernel = kernel or GaussianRBFKernel(bandwidth=1.0)
        self.bandwidth_range = bandwidth_range
        self.n_candidates = n_candidates
        self.optimizer = optimizer

        # Results storage
        self.optimal_bandwidth: Optional[float] = None
        self.optimal_error: Optional[float] = None
        self.search_history: List[Tuple[float, float]] = []

    def _compute_loocv_error(
        self,
        bandwidth: float,
        angles: np.ndarray,
        values: np.ndarray,
        distance_matrix: np.ndarray,
    ) -> float:
        """
        Compute LOOCV error for a given bandwidth.

        Uses precomputed distance matrix for efficiency.
        """
        n = len(values)
        self.kernel.bandwidth = bandwidth

        errors = np.zeros(n)

        for i in range(n):
            # Compute weights using precomputed distances
            weights = self.kernel(distance_matrix[i])
            weights[i] = 0  # Leave out point i

            weight_sum = np.sum(weights)

            if weight_sum < 1e-10:
                # Fallback: predict mean of other values
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                prediction = np.mean(values[mask])
            else:
                prediction = np.sum(weights * values) / weight_sum

            errors[i] = (prediction - values[i]) ** 2

        return np.mean(errors)

    def _compute_distance_matrix(self, angles: np.ndarray) -> np.ndarray:
        """
        Precompute pairwise distance matrix.
        """
        n = len(angles)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = great_circle_distance(
                    angles[i, 0], angles[i, 1],
                    angles[j, 0], angles[j, 1]
                )
                distances[i, j] = d
                distances[j, i] = d

        return distances

    def select(
        self,
        angles: np.ndarray,
        values: np.ndarray,
    ) -> float:
        """
        Select optimal bandwidth using LOOCV.

        Args:
            angles: Training angles as (N, 2) array of (theta, phi) pairs
            values: Training values as (N,) array

        Returns:
            Optimal bandwidth
        """
        angles = np.asarray(angles)
        values = np.asarray(values)

        if len(angles) != len(values):
            raise ValueError("angles and values must have same length")

        if len(angles) < 3:
            # Not enough points for meaningful LOOCV
            # Use rule of thumb instead
            return RuleOfThumbBandwidth.silverman(angles)

        # Precompute distance matrix
        distance_matrix = self._compute_distance_matrix(angles)

        # Define objective function
        def objective(h: float) -> float:
            error = self._compute_loocv_error(h, angles, values, distance_matrix)
            self.search_history.append((h, error))
            return error

        self.search_history = []

        if self.optimizer == "brent":
            # Use Brent's method for efficient optimization
            result = minimize_scalar(
                objective,
                bounds=self.bandwidth_range,
                method="bounded",
            )
            self.optimal_bandwidth = result.x
            self.optimal_error = result.fun

        else:  # grid search
            candidates = np.linspace(
                self.bandwidth_range[0],
                self.bandwidth_range[1],
                self.n_candidates
            )

            errors = [objective(h) for h in candidates]
            best_idx = np.argmin(errors)

            self.optimal_bandwidth = candidates[best_idx]
            self.optimal_error = errors[best_idx]

        return self.optimal_bandwidth

    def plot_search(
        self,
        ax=None,
        show: bool = True,
    ):
        """
        Plot the bandwidth search history.

        Args:
            ax: Matplotlib axis (creates new figure if None)
            show: Whether to call plt.show()
        """
        import matplotlib.pyplot as plt

        if not self.search_history:
            raise RuntimeError("No search history available. Run select() first.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        bandwidths, errors = zip(*sorted(self.search_history))

        ax.plot(bandwidths, errors, 'b-', linewidth=1.5, label='LOOCV Error')
        ax.scatter(bandwidths, errors, c='blue', s=20, alpha=0.5)

        if self.optimal_bandwidth is not None:
            ax.axvline(
                self.optimal_bandwidth,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Optimal h={self.optimal_bandwidth:.4f}'
            )

        ax.set_xlabel('Bandwidth (h)')
        ax.set_ylabel('LOO-CV Mean Squared Error')
        ax.set_title('Bandwidth Selection via LOOCV')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()


class RuleOfThumbBandwidth:
    """
    Rule-of-thumb bandwidth selection methods.

    These provide quick bandwidth estimates without cross-validation,
    useful as initial guesses or when data is limited.
    """

    @staticmethod
    def silverman(angles: np.ndarray) -> float:
        """
        Silverman's rule of thumb for bandwidth selection.

        h = 1.06 * σ * n^(-1/5)

        For spherical data, we use the average spread in angles.

        Args:
            angles: Training angles as (N, 2) array

        Returns:
            Bandwidth estimate
        """
        n = len(angles)

        # Estimate spread in each angular dimension
        sigma_theta = np.std(angles[:, 0])
        sigma_phi = np.std(angles[:, 1])

        # Use geometric mean of spreads
        sigma = np.sqrt(sigma_theta * sigma_phi)

        # Silverman's rule
        h = 1.06 * sigma * (n ** (-1/5))

        return max(h, 0.01)  # Ensure minimum bandwidth

    @staticmethod
    def scott(angles: np.ndarray) -> float:
        """
        Scott's rule of thumb for bandwidth selection.

        h = 3.5 * σ * n^(-1/3)

        Args:
            angles: Training angles as (N, 2) array

        Returns:
            Bandwidth estimate
        """
        n = len(angles)

        # Estimate spread
        sigma_theta = np.std(angles[:, 0])
        sigma_phi = np.std(angles[:, 1])
        sigma = np.sqrt(sigma_theta * sigma_phi)

        # Scott's rule
        h = 3.5 * sigma * (n ** (-1/3))

        return max(h, 0.01)

    @staticmethod
    def mean_nearest_neighbor(angles: np.ndarray, k: int = 5) -> float:
        """
        Estimate bandwidth from mean k-nearest-neighbor distance.

        Args:
            angles: Training angles as (N, 2) array
            k: Number of nearest neighbors to consider

        Returns:
            Bandwidth estimate
        """
        n = len(angles)
        k = min(k, n - 1)

        mean_knn_dist = 0.0

        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    d = great_circle_distance(
                        angles[i, 0], angles[i, 1],
                        angles[j, 0], angles[j, 1]
                    )
                    distances.append(d)

            distances.sort()
            mean_knn_dist += np.mean(distances[:k])

        mean_knn_dist /= n

        return mean_knn_dist


class GridSearchBandwidth:
    """
    Grid search bandwidth selection with custom error metric.

    This class allows using different error metrics beyond MSE
    for bandwidth selection.
    """

    def __init__(
        self,
        bandwidth_range: Tuple[float, float] = (0.01, 2.0),
        n_candidates: int = 30,
        kernel: Optional[Kernel] = None,
    ):
        """
        Initialize grid search selector.

        Args:
            bandwidth_range: Search range (min, max)
            n_candidates: Number of grid points
            kernel: Kernel function to use
        """
        self.bandwidth_range = bandwidth_range
        self.n_candidates = n_candidates
        self.kernel = kernel or GaussianRBFKernel(bandwidth=1.0)

        self.results: List[Tuple[float, float]] = []
        self.optimal_bandwidth: Optional[float] = None

    def select(
        self,
        angles: np.ndarray,
        values: np.ndarray,
        error_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> float:
        """
        Select bandwidth using grid search.

        Args:
            angles: Training angles (N, 2)
            values: Training values (N,)
            error_func: Custom error function(predictions, targets) -> error
                       Default: mean squared error

        Returns:
            Optimal bandwidth
        """
        if error_func is None:
            error_func = lambda pred, target: np.mean((pred - target) ** 2)

        candidates = np.linspace(
            self.bandwidth_range[0],
            self.bandwidth_range[1],
            self.n_candidates
        )

        self.results = []

        for h in candidates:
            estimator = NadarayaWatsonEstimator(kernel=self.kernel, bandwidth=h)
            estimator.fit(angles, values)

            # Leave-one-out predictions
            predictions = np.zeros(len(values))
            for i in range(len(values)):
                # Create estimator without point i
                mask = np.ones(len(values), dtype=bool)
                mask[i] = False
                temp_estimator = NadarayaWatsonEstimator(kernel=self.kernel, bandwidth=h)
                temp_estimator.fit(angles[mask], values[mask])
                predictions[i] = temp_estimator.predict(angles[i])

            error = error_func(predictions, values)
            self.results.append((h, error))

        best_idx = np.argmin([r[1] for r in self.results])
        self.optimal_bandwidth = self.results[best_idx][0]

        return self.optimal_bandwidth
