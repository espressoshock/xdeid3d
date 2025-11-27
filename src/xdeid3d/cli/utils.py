"""
CLI utility functions.

Helper functions for the command-line interface.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click

from xdeid3d import __version__

__all__ = [
    "setup_logging",
    "print_banner",
    "validate_path",
    "get_device",
    "format_size",
    "format_duration",
    "progress_callback",
    "ProgressBar",
]


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging for CLI.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logger = logging.getLogger("xdeid3d")
    logger.setLevel(getattr(logging, level.upper()))

    # Create console handler if not exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def print_banner() -> None:
    """Print CLI banner."""
    banner = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                       X-DeID3D                           ║
    ║     3D Explainability Framework for Face Anonymization   ║
    ║                      v{__version__:<10}                      ║
    ╚══════════════════════════════════════════════════════════╝
    """
    click.echo(click.style(banner, fg="cyan"))


def validate_path(
    path: Union[str, Path],
    must_exist: bool = True,
    must_be_dir: bool = False,
    must_be_file: bool = False,
    create_dirs: bool = False,
) -> Path:
    """
    Validate and resolve a file path.

    Args:
        path: Path to validate
        must_exist: Path must exist
        must_be_dir: Path must be a directory
        must_be_file: Path must be a file
        create_dirs: Create parent directories if needed

    Returns:
        Resolved Path object

    Raises:
        click.ClickException: If validation fails
    """
    path = Path(path).resolve()

    if must_exist and not path.exists():
        raise click.ClickException(f"Path does not exist: {path}")

    if must_be_dir and path.exists() and not path.is_dir():
        raise click.ClickException(f"Path is not a directory: {path}")

    if must_be_file and path.exists() and not path.is_file():
        raise click.ClickException(f"Path is not a file: {path}")

    if create_dirs and not path.exists():
        if must_be_file:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    return path


def get_device(device_str: Optional[str] = None) -> str:
    """
    Get the device to use for computation.

    Args:
        device_str: Device string (cpu, cuda, cuda:0, etc.)

    Returns:
        Validated device string
    """
    if device_str is None:
        # Auto-detect
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    # Validate device string
    device_str = device_str.lower()

    if device_str == "cpu":
        return "cpu"

    if device_str.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                click.echo("Warning: CUDA not available, using CPU", err=True)
                return "cpu"

            if ":" in device_str:
                device_idx = int(device_str.split(":")[1])
                if device_idx >= torch.cuda.device_count():
                    click.echo(
                        f"Warning: CUDA device {device_idx} not found, using cuda:0",
                        err=True
                    )
                    return "cuda:0"

            return device_str

        except ImportError:
            click.echo("Warning: PyTorch not installed, using CPU", err=True)
            return "cpu"

    click.echo(f"Warning: Unknown device '{device_str}', using CPU", err=True)
    return "cpu"


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration to human readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"

    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"


def progress_callback(current: int, total: int, message: str = "") -> None:
    """
    Simple progress callback for operations.

    Args:
        current: Current progress value
        total: Total progress value
        message: Optional message
    """
    pct = current / total * 100 if total > 0 else 0
    bar_width = 40
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_width - filled)

    click.echo(f"\r[{bar}] {pct:5.1f}% {message}", nl=False)

    if current >= total:
        click.echo()  # New line at end


class ProgressBar:
    """
    Context manager for progress bar display.

    Example:
        >>> with ProgressBar(total=100, description="Processing") as bar:
        ...     for i in range(100):
        ...         bar.update(1)
    """

    def __init__(
        self,
        total: int,
        description: str = "",
        unit: str = "it",
        disable: bool = False,
    ):
        self.total = total
        self.description = description
        self.unit = unit
        self.disable = disable
        self.current = 0
        self._bar = None

    def __enter__(self) -> "ProgressBar":
        if not self.disable:
            try:
                from tqdm import tqdm
                self._bar = tqdm(
                    total=self.total,
                    desc=self.description,
                    unit=self.unit,
                )
            except ImportError:
                # Fallback to simple progress
                self._bar = None
                if self.description:
                    click.echo(f"{self.description}...")
        return self

    def __exit__(self, *args) -> None:
        if self._bar is not None:
            self._bar.close()
        elif not self.disable and self.current > 0:
            click.echo(f" Done. ({self.current} {self.unit})")

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current += n
        if self._bar is not None:
            self._bar.update(n)
        elif not self.disable:
            progress_callback(self.current, self.total)

    def set_description(self, desc: str) -> None:
        """Update description."""
        self.description = desc
        if self._bar is not None:
            self._bar.set_description(desc)


def parse_metrics(metrics_str: str) -> List[str]:
    """
    Parse metrics string into list of metric names.

    Args:
        metrics_str: Comma-separated metric names or preset name

    Returns:
        List of metric names
    """
    if metrics_str == "standard":
        return ["arcface_cosine_distance", "psnr", "ssim"]
    elif metrics_str == "full":
        return [
            "arcface_cosine_distance",
            "psnr",
            "ssim",
            "lpips",
            "face_confidence",
        ]
    else:
        return [m.strip() for m in metrics_str.split(",")]


def parse_seeds(seeds_str: str) -> List[int]:
    """
    Parse seeds string into list of integers.

    Args:
        seeds_str: Seeds as "0,1,2" or "0-5"

    Returns:
        List of seed integers
    """
    if "-" in seeds_str:
        start, end = seeds_str.split("-")
        return list(range(int(start), int(end) + 1))
    else:
        return [int(s.strip()) for s in seeds_str.split(",")]


def create_output_dir(
    base_dir: Union[str, Path],
    experiment_name: Optional[str] = None,
    timestamp: bool = True,
) -> Path:
    """
    Create output directory for experiment.

    Args:
        base_dir: Base output directory
        experiment_name: Optional experiment name
        timestamp: Whether to add timestamp

    Returns:
        Created directory path
    """
    from datetime import datetime

    base_dir = Path(base_dir)

    if experiment_name:
        dir_name = experiment_name
    else:
        dir_name = "experiment"

    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{dir_name}_{ts}"

    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir
