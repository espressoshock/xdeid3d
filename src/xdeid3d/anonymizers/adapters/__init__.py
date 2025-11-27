"""
Built-in anonymizer adapters.

This module provides baseline anonymizers for testing and comparison.
All adapters are automatically registered with the AnonymizerRegistry.
"""

from xdeid3d.anonymizers.adapters.blur import BlurAnonymizer
from xdeid3d.anonymizers.adapters.pixelate import PixelateAnonymizer
from xdeid3d.anonymizers.adapters.blackout import BlackoutAnonymizer
from xdeid3d.anonymizers.adapters.identity import IdentityAnonymizer

__all__ = [
    "BlurAnonymizer",
    "PixelateAnonymizer",
    "BlackoutAnonymizer",
    "IdentityAnonymizer",
]