"""
Experiments viewer for X-DeID3D.

Flask-based web application for browsing and comparing
evaluation experiments and results.
"""

from xdeid3d.viewer.app import create_app, ViewerConfig

__all__ = ["create_app", "ViewerConfig"]
