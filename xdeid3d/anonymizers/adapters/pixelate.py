"""
Pixelation anonymizer.

A simple baseline anonymizer that applies pixelation to images.
"""

from typing import Any, Optional
import numpy as np

from xdeid3d.anonymizers.base import BaseAnonymizer, AnonymizationResult
from xdeid3d.anonymizers.registry import AnonymizerRegistry

__all__ = ["PixelateAnonymizer"]


@AnonymizerRegistry.register("pixelate")
class PixelateAnonymizer(BaseAnonymizer):
    """
    Pixelation anonymizer.

    Applies pixelation effect by downscaling and upscaling the image,
    creating a blocky/mosaic appearance.

    Args:
        block_size: Size of each pixel block (larger = more anonymization)
        face_only: If True, only pixelate detected face region
        padding: Extra padding around face bbox when face_only=True

    Example:
        >>> anonymizer = PixelateAnonymizer(block_size=10)
        >>> result = anonymizer.anonymize(image)
    """

    def __init__(
        self,
        block_size: int = 10,
        face_only: bool = False,
        padding: int = 10,
        **kwargs: Any,
    ):
        super().__init__(name="Pixelate", **kwargs)
        self.block_size = max(2, block_size)
        self.face_only = face_only
        self.padding = padding

    def _pixelate_region(
        self, region: np.ndarray, block_size: int
    ) -> np.ndarray:
        """Apply pixelation to a region."""
        import cv2

        h, w = region.shape[:2]

        # Calculate small size
        small_h = max(1, h // block_size)
        small_w = max(1, w // block_size)

        # Downscale then upscale
        small = cv2.resize(
            region, (small_w, small_h), interpolation=cv2.INTER_LINEAR
        )
        pixelated = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_NEAREST
        )

        return pixelated

    def _anonymize_single(
        self, image: np.ndarray, **kwargs: Any
    ) -> AnonymizationResult:
        """Apply pixelation to the image."""
        # Get optional parameters
        block_size = kwargs.get("block_size", self.block_size)
        face_bbox = kwargs.get("face_bbox")

        if self.face_only and face_bbox is not None:
            # Pixelate only face region
            x1, y1, x2, y2 = face_bbox
            # Add padding
            h, w = image.shape[:2]
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w, x2 + self.padding)
            y2 = min(h, y2 + self.padding)

            # Copy image and pixelate face region
            result = image.copy()
            face_region = image[y1:y2, x1:x2]
            pixelated_face = self._pixelate_region(face_region, block_size)
            result[y1:y2, x1:x2] = pixelated_face

            return AnonymizationResult(
                anonymized_image=result,
                face_detected=True,
                face_bbox=(x1, y1, x2, y2),
            )
        else:
            # Pixelate entire image
            pixelated = self._pixelate_region(image, block_size)
            return AnonymizationResult(
                anonymized_image=pixelated,
                face_detected=True,
            )
