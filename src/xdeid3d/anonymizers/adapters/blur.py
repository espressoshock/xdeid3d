"""
Gaussian blur anonymizer.

A simple baseline anonymizer that applies Gaussian blur to detected faces.
"""

from typing import Any, Optional
import numpy as np

from xdeid3d.anonymizers.base import BaseAnonymizer, AnonymizationResult
from xdeid3d.anonymizers.registry import AnonymizerRegistry

__all__ = ["BlurAnonymizer"]


@AnonymizerRegistry.register("blur")
class BlurAnonymizer(BaseAnonymizer):
    """
    Gaussian blur anonymizer.

    Applies Gaussian blur to the entire image. For face-specific blur,
    use with a face detection component.

    Args:
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation of the Gaussian kernel
        face_only: If True, only blur detected face region (requires face detector)
        padding: Extra padding around face bbox when face_only=True

    Example:
        >>> anonymizer = BlurAnonymizer(kernel_size=51, sigma=20.0)
        >>> result = anonymizer.anonymize(image)
    """

    def __init__(
        self,
        kernel_size: int = 51,
        sigma: float = 20.0,
        face_only: bool = False,
        padding: int = 10,
        **kwargs: Any,
    ):
        super().__init__(name="Blur", **kwargs)
        # Ensure kernel size is odd
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.face_only = face_only
        self.padding = padding
        self._face_detector = None

    def _anonymize_single(
        self, image: np.ndarray, **kwargs: Any
    ) -> AnonymizationResult:
        """Apply Gaussian blur to the image."""
        import cv2

        # Get optional parameters
        kernel_size = kwargs.get("kernel_size", self.kernel_size)
        sigma = kwargs.get("sigma", self.sigma)
        face_bbox = kwargs.get("face_bbox")

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        if self.face_only and face_bbox is not None:
            # Blur only face region
            x1, y1, x2, y2 = face_bbox
            # Add padding
            h, w = image.shape[:2]
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w, x2 + self.padding)
            y2 = min(h, y2 + self.padding)

            # Copy image and blur face region
            result = image.copy()
            face_region = image[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(
                face_region, (kernel_size, kernel_size), sigma
            )
            result[y1:y2, x1:x2] = blurred_face

            return AnonymizationResult(
                anonymized_image=result,
                face_detected=True,
                face_bbox=(x1, y1, x2, y2),
            )
        else:
            # Blur entire image
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            return AnonymizationResult(
                anonymized_image=blurred,
                face_detected=True,
            )
