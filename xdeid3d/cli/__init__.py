"""
X-DeID3D Command Line Interface.

This package provides the command-line interface for X-DeID3D,
including commands for evaluation, generation, and benchmarking.

Usage:
    xdeid3d --help
    xdeid3d evaluate --help
    xdeid3d generate --help
"""

from xdeid3d.cli.app import cli, main

__all__ = ["cli", "main"]
