"""
Core modules for X-DeID3D.

This package provides the mathematical foundations for spherical
kernel regression and 3D synthesis.

Subpackages:
    geometry: Spherical geometry utilities (great-circle distance, conversions)
    regression: Nadaraya-Watson kernel regression with LOOCV bandwidth selection
    synthesis: 3D head synthesis engine (SphereHead integration)
"""

from xdeid3d.core import geometry
from xdeid3d.core import regression

__all__ = ["geometry", "regression"]