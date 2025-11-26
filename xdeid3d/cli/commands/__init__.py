"""
CLI command modules.

Subcommand groups for the X-DeID3D CLI.
"""

from xdeid3d.cli.commands.generate import generate
from xdeid3d.cli.commands.evaluate import evaluate
from xdeid3d.cli.commands.benchmark import benchmark

__all__ = ["generate", "evaluate", "benchmark"]
