"""
GUI application for X-DeID3D.

FastAPI-based web application for interactive evaluation
and visualization of face anonymization systems.
"""

from xdeid3d.gui.app import create_app, GUIConfig, run_gui

__all__ = ["create_app", "GUIConfig", "run_gui"]
