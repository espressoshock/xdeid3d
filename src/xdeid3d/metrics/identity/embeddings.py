"""
Face embedding extraction and caching.

This module provides utilities for extracting face embeddings using
insightface or custom models, with optional caching for performance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable
import hashlib
import numpy as np

__all__ = [
    "EmbeddingExtractor",
    "InsightFaceExtractor",
    "EmbeddingCache",
    "FaceDetectionResult",
]


@dataclass
class FaceDetectionResult:
    """Result of face detection."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    landmarks: Optional[np.ndarray] = None
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None


@runtime_checkable
class EmbeddingExtractor(Protocol):
    """Protocol for face embedding extractors."""

    def extract(
        self, image: np.ndarray, **kwargs: Any
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.

        Args:
            image: Input image (H, W, C) in RGB format

        Returns:
            Embedding vector or None if no face detected
        """
        ...

    def detect_and_extract(
        self, image: np.ndarray, **kwargs: Any
    ) -> Tuple[Optional[FaceDetectionResult], Optional[np.ndarray]]:
        """
        Detect face and extract embedding.

        Returns:
            Tuple of (detection_result, embedding), both None if no face
        """
        ...


class InsightFaceExtractor:
    """
    Face embedding extractor using insightface library.

    Uses the buffalo_l model by default, which provides good accuracy
    for identity verification.

    Example:
        >>> extractor = InsightFaceExtractor()
        >>> embedding = extractor.extract(image)
        >>> distance = cosine_distance(emb1, emb2)
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cpu",
        det_size: Tuple[int, int] = (640, 640),
    ):
        """
        Initialize InsightFace extractor.

        Args:
            model_name: Model pack name (e.g., 'buffalo_l', 'buffalo_sc')
            device: Device to use ('cpu' or 'cuda')
            det_size: Detection input size
        """
        self.model_name = model_name
        self.device = device
        self.det_size = det_size
        self._app = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of the model."""
        if self._initialized:
            return

        try:
            from insightface.app import FaceAnalysis

            # Set providers based on device
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            self._app = FaceAnalysis(
                name=self.model_name,
                providers=providers,
            )
            self._app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=self.det_size)
            self._initialized = True

        except ImportError:
            raise ImportError(
                "insightface is required for identity metrics. "
                "Install with: pip install insightface onnxruntime"
            )

    def extract(
        self, image: np.ndarray, **kwargs: Any
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            512-d embedding vector or None if no face detected
        """
        self._ensure_initialized()

        # InsightFace expects BGR
        image_bgr = image[:, :, ::-1].copy()

        faces = self._app.get(image_bgr)

        if not faces:
            return None

        # Return embedding of largest face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.embedding

    def detect_and_extract(
        self, image: np.ndarray, **kwargs: Any
    ) -> Tuple[Optional[FaceDetectionResult], Optional[np.ndarray]]:
        """
        Detect face and extract embedding.

        Returns:
            Tuple of (detection_result, embedding)
        """
        self._ensure_initialized()

        image_bgr = image[:, :, ::-1].copy()
        faces = self._app.get(image_bgr)

        if not faces:
            return None, None

        # Get largest face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        detection = FaceDetectionResult(
            bbox=tuple(map(int, largest.bbox)),
            landmarks=largest.landmark_2d_106 if hasattr(largest, "landmark_2d_106") else None,
            confidence=float(largest.det_score) if hasattr(largest, "det_score") else 1.0,
            embedding=largest.embedding,
        )

        return detection, largest.embedding

    def extract_batch(
        self, images: List[np.ndarray], **kwargs: Any
    ) -> List[Optional[np.ndarray]]:
        """Extract embeddings from multiple images."""
        return [self.extract(img, **kwargs) for img in images]


class EmbeddingCache:
    """
    Cache for face embeddings to avoid recomputation.

    Caches embeddings by image hash, allowing fast lookup for
    repeated queries on the same images.

    Example:
        >>> cache = EmbeddingCache()
        >>> emb = cache.get_or_compute(image, extractor)
    """

    def __init__(
        self,
        max_size: int = 10000,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            persist_path: Path to persist cache (optional)
        """
        self.max_size = max_size
        self.persist_path = persist_path
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []

        if persist_path and persist_path.exists():
            self._load_cache()

    def _compute_hash(self, image: np.ndarray) -> str:
        """Compute hash for image."""
        # Use a fast hash of downsampled image
        small = image[::8, ::8].tobytes()
        return hashlib.md5(small).hexdigest()

    def get(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get cached embedding for image."""
        key = self._compute_hash(image)

        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        return None

    def put(self, image: np.ndarray, embedding: np.ndarray) -> None:
        """Cache embedding for image."""
        key = self._compute_hash(image)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = embedding
        self._access_order.append(key)

    def get_or_compute(
        self,
        image: np.ndarray,
        extractor: EmbeddingExtractor,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        """
        Get cached embedding or compute and cache.

        Args:
            image: Input image
            extractor: Extractor to use if not cached

        Returns:
            Embedding or None if extraction fails
        """
        cached = self.get(image)
        if cached is not None:
            return cached

        embedding = extractor.extract(image, **kwargs)
        if embedding is not None:
            self.put(image, embedding)

        return embedding

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.persist_path and self.persist_path.exists():
            try:
                data = np.load(self.persist_path, allow_pickle=True)
                self._cache = dict(data.item())
            except Exception:
                pass

    def save(self) -> None:
        """Save cache to disk."""
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.persist_path, self._cache)

    @property
    def size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)

    def __len__(self) -> int:
        return self.size


def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine distance between two embeddings.

    Cosine distance = 1 - cosine_similarity

    Args:
        emb1: First embedding vector
        emb2: Second embedding vector

    Returns:
        Distance in range [0, 2], where 0 = identical, 2 = opposite
    """
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
    similarity = np.dot(emb1, emb2)
    return float(1 - similarity)


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute Euclidean distance between two embeddings."""
    return float(np.linalg.norm(emb1 - emb2))
