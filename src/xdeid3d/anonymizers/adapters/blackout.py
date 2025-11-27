"""
Blackout anonymizer.

A simple baseline anonymizer that replaces face regions with solid color.
"""

from typing import Any, Optional, Tuple
import numpy as np

from xdeid3d.anonymizers.base import BaseAnonymizer, AnonymizationResult
from xdeid3d.anonymizers.registry import AnonymizerRegistry

__all__ = ["BlackoutAnonymizer"]


@AnonymizerRegistry.register("blackout")
class BlackoutAnonymizer(BaseAnonymizer):
    """
    Blackout anonymizer.

    Replaces face regions with a solid color (default black).

    Args:
        color: RGB color tuple for the blackout (default: black)
        face_only: If True, requires face bbox to be provided
        padding: Extra padding around face bbox

    Example:
        >>> anonymizer = BlackoutAnonymizer(color=(0, 0, 0))
        >>> result = anonymizer.anonymize(image, face_bbox=(100, 100, 200, 200))
    """

    def __init__(
        self,
        color: Tuple[int, int, int] = (0, 0, 0),
        face_only: bool = True,
        padding: int = 10,
        **kwargs: Any,
    ):
        super().__init__(name="Blackout", **kwargs)
        self.color = color
        self.face_only = face_only
        self.padding = padding

    def _anonymize_single(
        self, image: np.ndarray, **kwargs: Any
    ) -> AnonymizationResult:
        """Apply blackout to the image."""
        face_bbox = kwargs.get("face_bbox")
        color = kwargs.get("color", self.color)

        if self.face_only and face_bbox is None:
            # No face bbox provided, return original with warning
            return AnonymizationResult(
                anonymized_image=image.copy(),
                face_detected=False,
                metadata={"warning": "No face_bbox provided for blackout"},
            )

        result = image.copy()

        if face_bbox is not None:
            # Blackout face region
            x1, y1, x2, y2 = face_bbox
            # Add padding
            h, w = image.shape[:2]
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w, x2 + self.padding)
            y2 = min(h, y2 + self.padding)

            result[y1:y2, x1:x2] = color

            return AnonymizationResult(
                anonymized_image=result,
                face_detected=True,
                face_bbox=(x1, y1, x2, y2),
            )
        else:
            # Blackout entire image
            result[:] = color
            return AnonymizationResult(
                anonymized_image=result,
                face_detected=True,
            )
