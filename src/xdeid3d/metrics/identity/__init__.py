"""
Identity metrics for face anonymization evaluation.

This module provides metrics for measuring identity change, including
ArcFace-based cosine distance and binary identity change indicators.
"""

from xdeid3d.metrics.identity.embeddings import (
    EmbeddingExtractor,
    InsightFaceExtractor,
    EmbeddingCache,
    FaceDetectionResult,
    cosine_distance,
    euclidean_distance,
)
from xdeid3d.metrics.identity.arcface import (
    ArcFaceCosineDistance,
    IdentityChangeMetric,
)

__all__ = [
    # Embedding utilities
    "EmbeddingExtractor",
    "InsightFaceExtractor",
    "EmbeddingCache",
    "FaceDetectionResult",
    "cosine_distance",
    "euclidean_distance",
    # Metrics
    "ArcFaceCosineDistance",
    "IdentityChangeMetric",
]