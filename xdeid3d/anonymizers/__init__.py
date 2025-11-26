"""
Pluggable anonymizer framework for X-DeID3D.

This package provides the interface and utilities for face anonymization,
allowing users to plug in any anonymization method via the AnonymizerProtocol.

Example:
    >>> from xdeid3d.anonymizers import AnonymizerRegistry, BaseAnonymizer
    >>>
    >>> @AnonymizerRegistry.register("my_method")
    ... class MyAnonymizer(BaseAnonymizer):
    ...     def _anonymize_single(self, image, **kwargs):
    ...         # Your logic here
    ...         return AnonymizationResult(anonymized_image=image)
    >>>
    >>> anonymizer = AnonymizerRegistry.get("my_method")
"""

from xdeid3d.anonymizers.base import (
    AnonymizerProtocol,
    BaseAnonymizer,
    AnonymizationResult,
    BatchAnonymizationResult,
)
from xdeid3d.anonymizers.registry import (
    AnonymizerRegistry,
    register_anonymizer,
    create_anonymizer,
)

__all__ = [
    # Base classes
    "AnonymizerProtocol",
    "BaseAnonymizer",
    "AnonymizationResult",
    "BatchAnonymizationResult",
    # Registry
    "AnonymizerRegistry",
    "register_anonymizer",
    "create_anonymizer",
]