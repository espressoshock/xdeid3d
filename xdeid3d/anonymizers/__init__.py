"""
Pluggable anonymizer framework for X-DeID3D.

This package provides the interface and utilities for face anonymization,
allowing users to plug in any anonymization method via the AnonymizerProtocol.
"""

from xdeid3d.anonymizers.base import (
    AnonymizerProtocol,
    BaseAnonymizer,
    AnonymizationResult,
    BatchAnonymizationResult,
)

__all__ = [
    "AnonymizerProtocol",
    "BaseAnonymizer",
    "AnonymizationResult",
    "BatchAnonymizationResult",
]