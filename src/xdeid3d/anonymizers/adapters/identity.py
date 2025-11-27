"""
Identity (no-op) anonymizer.

A baseline that returns the input unchanged, useful for testing and comparison.
"""

from typing import Any
import numpy as np

from xdeid3d.anonymizers.base import BaseAnonymizer, AnonymizationResult
from xdeid3d.anonymizers.registry import AnonymizerRegistry

__all__ = ["IdentityAnonymizer"]


@AnonymizerRegistry.register("identity")
@AnonymizerRegistry.register("none")
@AnonymizerRegistry.register("noop")
class IdentityAnonymizer(BaseAnonymizer):
    """
    Identity (no-op) anonymizer.

    Returns the input image unchanged. Useful for:
    - Testing the evaluation pipeline
    - Baseline comparisons
    - Debugging

    Example:
        >>> anonymizer = IdentityAnonymizer()
        >>> result = anonymizer.anonymize(image)
        >>> assert np.array_equal(result.anonymized_image, image)
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="Identity", **kwargs)

    def _anonymize_single(
        self, image: np.ndarray, **kwargs: Any
    ) -> AnonymizationResult:
        """Return the image unchanged."""
        return AnonymizationResult(
            anonymized_image=image.copy(),
            face_detected=True,
            metadata={"anonymizer": "identity (no-op)"},
        )
