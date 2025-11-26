"""
Base anonymizer protocol and classes.

This module defines the protocol (interface) that all anonymizers must
implement, along with base classes and result containers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, TypeVar, runtime_checkable
import numpy as np

__all__ = [
    "AnonymizerProtocol",
    "BaseAnonymizer",
    "AnonymizationResult",
    "BatchAnonymizationResult",
]


@dataclass
class AnonymizationResult:
    """
    Result of anonymizing a single image.

    Attributes:
        anonymized_image: The anonymized image as numpy array (H, W, C)
        face_detected: Whether a face was detected in the input
        face_bbox: Bounding box of detected face (x1, y1, x2, y2) or None
        face_landmarks: Facial landmarks if available
        confidence: Detection confidence score
        processing_time_ms: Time taken to process in milliseconds
        metadata: Additional anonymizer-specific metadata
    """

    anonymized_image: np.ndarray
    face_detected: bool = True
    face_bbox: Optional[tuple[int, int, int, int]] = None
    face_landmarks: Optional[np.ndarray] = None
    confidence: float = 1.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if anonymization was successful."""
        return self.anonymized_image is not None and self.face_detected


@dataclass
class BatchAnonymizationResult:
    """
    Result of anonymizing a batch of images.

    Attributes:
        results: List of individual anonymization results
        total_time_ms: Total processing time in milliseconds
        batch_size: Number of images in the batch
    """

    results: List[AnonymizationResult]
    total_time_ms: float = 0.0

    @property
    def batch_size(self) -> int:
        return len(self.results)

    @property
    def images(self) -> List[np.ndarray]:
        """Get all anonymized images."""
        return [r.anonymized_image for r in self.results]

    @property
    def success_rate(self) -> float:
        """Fraction of successfully processed images."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx: int) -> AnonymizationResult:
        return self.results[idx]


@runtime_checkable
class AnonymizerProtocol(Protocol):
    """
    Protocol defining the interface for face anonymizers.

    Any class implementing this protocol can be used as an anonymizer
    in the X-DeID3D evaluation pipeline.

    Example:
        >>> class MyAnonymizer:
        ...     def anonymize(self, image: np.ndarray, **kwargs) -> AnonymizationResult:
        ...         # Your anonymization logic here
        ...         return AnonymizationResult(anonymized_image=processed)
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyAnonymizer"
        >>>
        >>> # Use in evaluation
        >>> anonymizer = MyAnonymizer()
        >>> result = anonymizer.anonymize(image)
    """

    @property
    def name(self) -> str:
        """Return the anonymizer name."""
        ...

    def anonymize(
        self, image: np.ndarray, **kwargs: Any
    ) -> AnonymizationResult:
        """
        Anonymize a single image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format,
                   with values in range [0, 255] (uint8) or [0, 1] (float)
            **kwargs: Additional anonymizer-specific parameters

        Returns:
            AnonymizationResult containing the anonymized image and metadata
        """
        ...


T = TypeVar("T", bound="BaseAnonymizer")


class BaseAnonymizer(ABC):
    """
    Abstract base class for face anonymizers.

    This class provides a template for implementing anonymizers with
    common functionality like batch processing, input validation,
    and timing.

    Subclasses must implement:
        - _anonymize_single: Core anonymization logic for a single image

    Example:
        >>> class BlurAnonymizer(BaseAnonymizer):
        ...     def __init__(self, kernel_size: int = 51):
        ...         super().__init__(name="Blur")
        ...         self.kernel_size = kernel_size
        ...
        ...     def _anonymize_single(self, image: np.ndarray, **kwargs) -> AnonymizationResult:
        ...         import cv2
        ...         blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        ...         return AnonymizationResult(anonymized_image=blurred)
    """

    def __init__(
        self,
        name: str = "BaseAnonymizer",
        batch_size: int = 1,
        device: Optional[str] = None,
    ):
        """
        Initialize base anonymizer.

        Args:
            name: Name identifier for the anonymizer
            batch_size: Default batch size for batch processing
            device: Device to use (e.g., 'cuda', 'cpu')
        """
        self._name = name
        self._batch_size = batch_size
        self._device = device
        self._initialized = False

    @property
    def name(self) -> str:
        """Return the anonymizer name."""
        return self._name

    @property
    def batch_size(self) -> int:
        """Return the default batch size."""
        return self._batch_size

    @property
    def device(self) -> Optional[str]:
        """Return the device being used."""
        return self._device

    def initialize(self) -> None:
        """
        Initialize the anonymizer (load models, etc.).

        Override this method if your anonymizer needs initialization.
        Called automatically on first use if not called explicitly.
        """
        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure the anonymizer is initialized."""
        if not self._initialized:
            self.initialize()

    @abstractmethod
    def _anonymize_single(
        self, image: np.ndarray, **kwargs: Any
    ) -> AnonymizationResult:
        """
        Core anonymization logic for a single image.

        Subclasses must implement this method.

        Args:
            image: Normalized input image (H, W, C), uint8, RGB
            **kwargs: Additional parameters

        Returns:
            AnonymizationResult
        """
        pass

    def anonymize(
        self, image: np.ndarray, **kwargs: Any
    ) -> AnonymizationResult:
        """
        Anonymize a single image.

        Args:
            image: Input image as numpy array (H, W, C)
            **kwargs: Additional anonymizer-specific parameters

        Returns:
            AnonymizationResult containing the anonymized image
        """
        import time

        self._ensure_initialized()

        # Validate and normalize input
        image = self._validate_input(image)

        # Time the operation
        start_time = time.perf_counter()
        result = self._anonymize_single(image, **kwargs)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result.processing_time_ms = elapsed_ms
        return result

    def anonymize_batch(
        self, images: Sequence[np.ndarray], **kwargs: Any
    ) -> BatchAnonymizationResult:
        """
        Anonymize a batch of images.

        Default implementation processes images sequentially.
        Override for optimized batch processing.

        Args:
            images: Sequence of input images
            **kwargs: Additional parameters passed to each anonymization

        Returns:
            BatchAnonymizationResult containing all results
        """
        import time

        self._ensure_initialized()

        start_time = time.perf_counter()
        results = []

        for image in images:
            result = self.anonymize(image, **kwargs)
            results.append(result)

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return BatchAnonymizationResult(
            results=results,
            total_time_ms=total_time_ms,
        )

    def _validate_input(self, image: np.ndarray) -> np.ndarray:
        """
        Validate and normalize input image.

        Args:
            image: Input image

        Returns:
            Normalized image (uint8, RGB, HWC)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")

        if image.ndim == 2:
            # Grayscale to RGB
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")

        if image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        elif image.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[2]}")

        # Normalize to uint8 if float
        if image.dtype in (np.float32, np.float64):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"

    def __call__(
        self, image: np.ndarray, **kwargs: Any
    ) -> AnonymizationResult:
        """Allow calling the anonymizer directly."""
        return self.anonymize(image, **kwargs)
