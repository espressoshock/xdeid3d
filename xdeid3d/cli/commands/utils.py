"""
Utility commands for X-DeID3D CLI.

Miscellaneous utility commands for data preparation,
format conversion, and system operations.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import click

from xdeid3d.cli.utils import (
    validate_path,
    get_device,
    ProgressBar,
    format_size,
)


@click.group(name="utils")
@click.pass_context
def utils(ctx: click.Context) -> None:
    """Utility commands.

    \b
    Commands:
      list-anonymizers   List registered anonymizers
      list-metrics       List available metrics
      convert            Convert between file formats
      extract-frames     Extract frames from video
      prepare-dataset    Prepare dataset for evaluation
      check-models       Check model availability
      clear-cache        Clear cached embeddings and data
    """
    pass


@utils.command("list-anonymizers")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed information"
)
@click.pass_context
def list_anonymizers(ctx: click.Context, verbose: bool) -> None:
    """List all registered anonymizers."""
    from xdeid3d.anonymizers import list_anonymizers as get_anonymizers

    click.echo("\nRegistered Anonymizers")
    click.echo("=" * 40)

    anonymizers = get_anonymizers()

    if not anonymizers:
        click.echo("No anonymizers registered.")
        click.echo("\nTo register an anonymizer, use:")
        click.echo("  from xdeid3d.anonymizers import register_anonymizer")
        return

    for name, info in sorted(anonymizers.items()):
        click.echo(f"\n{name}")

        if verbose:
            if "description" in info:
                click.echo(f"  Description: {info['description']}")
            if "version" in info:
                click.echo(f"  Version: {info['version']}")
            if "requires_gpu" in info:
                click.echo(f"  Requires GPU: {info['requires_gpu']}")
            if "model_path" in info:
                click.echo(f"  Model: {info['model_path']}")


@utils.command("list-metrics")
@click.option(
    "--category",
    type=click.Choice(["all", "identity", "quality", "temporal", "explainability"]),
    default="all",
    help="Filter by category"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed information"
)
@click.pass_context
def list_metrics(ctx: click.Context, category: str, verbose: bool) -> None:
    """List all available metrics."""
    from xdeid3d.metrics import list_metrics as get_metrics

    click.echo("\nAvailable Metrics")
    click.echo("=" * 40)

    metrics = get_metrics()

    for name, info in sorted(metrics.items()):
        metric_category = info.get("category", "unknown")

        if category != "all" and metric_category != category:
            continue

        click.echo(f"\n{name}")
        click.echo(f"  Category: {metric_category}")

        if verbose:
            if "description" in info:
                click.echo(f"  Description: {info['description']}")
            if "range" in info:
                click.echo(f"  Range: {info['range']}")
            if "higher_is_better" in info:
                click.echo(f"  Higher is better: {info['higher_is_better']}")


@utils.command("convert")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--format",
    type=click.Choice(["json", "npz", "csv", "ply"]),
    default=None,
    help="Output format (auto-detected from extension)"
)
@click.pass_context
def convert(
    ctx: click.Context,
    input_file: str,
    output_file: str,
    format: Optional[str],
) -> None:
    """
    Convert between file formats.

    INPUT_FILE: Source file
    OUTPUT_FILE: Destination file

    Supported conversions:
    - JSON <-> NPZ (evaluation results)
    - JSON -> CSV (tabular export)
    - NPZ -> PLY (heatmap to mesh)
    """
    import json
    import numpy as np

    input_path = validate_path(input_file, must_exist=True, must_be_file=True)
    output_path = Path(output_file)

    # Determine output format
    if format is None:
        format = output_path.suffix.lstrip(".").lower()

    input_format = input_path.suffix.lstrip(".").lower()

    click.echo(f"Converting {input_format} -> {format}")

    # JSON to NPZ
    if input_format == "json" and format == "npz":
        with open(input_path) as f:
            data = json.load(f)

        # Extract frame metrics for conversion
        if "frame_metrics" in data:
            frame_data = data["frame_metrics"]
        elif isinstance(data, list):
            frame_data = data
        else:
            raise click.ClickException("Unknown JSON format")

        # Convert to arrays
        arrays = {}
        for key in frame_data[0].keys():
            values = [d.get(key) for d in frame_data]
            if all(isinstance(v, (int, float)) for v in values if v is not None):
                arrays[key] = np.array([v if v is not None else np.nan for v in values])

        np.savez(output_path, **arrays)

    # NPZ to JSON
    elif input_format == "npz" and format == "json":
        data = np.load(input_path)
        result = {}

        for key in data.files:
            arr = data[key]
            if arr.ndim == 1:
                result[key] = arr.tolist()
            else:
                result[key] = arr.tolist()

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    # JSON to CSV
    elif input_format == "json" and format == "csv":
        import csv

        with open(input_path) as f:
            data = json.load(f)

        if "frame_metrics" in data:
            frame_data = data["frame_metrics"]
        elif isinstance(data, list):
            frame_data = data
        else:
            raise click.ClickException("Unknown JSON format")

        if not frame_data:
            raise click.ClickException("No data to convert")

        headers = list(frame_data[0].keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(frame_data)

    # NPZ to PLY (heatmap to colored mesh)
    elif input_format == "npz" and format == "ply":
        from xdeid3d.visualization import MeshExporter

        data = np.load(input_path)

        # Check for required arrays
        yaws = data.get("yaws", data.get("yaw", None))
        pitches = data.get("pitches", data.get("pitch", None))
        scores = data.get("scores", data.get("score", None))

        if scores is None:
            raise click.ClickException("No score data found in NPZ file")

        # Create mesh exporter
        exporter = MeshExporter()

        if yaws is not None and pitches is not None:
            for yaw, pitch, score in zip(yaws.flat, pitches.flat, scores.flat):
                exporter.add_score(float(yaw), float(pitch), float(score))
        else:
            # Assume grid data
            ny, nx = scores.shape
            for i in range(ny):
                for j in range(nx):
                    yaw = (j / nx) * 2 * np.pi - np.pi
                    pitch = (i / ny) * np.pi
                    exporter.add_score(yaw, pitch, float(scores[i, j]))

        # Create sphere mesh and export
        vertices, faces = _create_sphere_mesh()
        exporter.export_ply(str(output_path), vertices, faces)

    else:
        raise click.ClickException(f"Unsupported conversion: {input_format} -> {format}")

    click.echo(f"Converted: {output_path}")


@utils.command("extract-frames")
@click.argument("video_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for frames"
)
@click.option(
    "--format",
    type=click.Choice(["png", "jpg"]),
    default="png",
    help="Output image format"
)
@click.option(
    "--sample-rate",
    type=int,
    default=1,
    help="Extract every Nth frame (default: 1)"
)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Maximum frames to extract"
)
@click.option(
    "--resize",
    type=str,
    default=None,
    help="Resize frames to WxH"
)
@click.pass_context
def extract_frames(
    ctx: click.Context,
    video_file: str,
    output_dir: Optional[str],
    format: str,
    sample_rate: int,
    max_frames: Optional[int],
    resize: Optional[str],
) -> None:
    """
    Extract frames from video file.

    VIDEO_FILE: Path to video file
    """
    import cv2

    video_path = validate_path(video_file, must_exist=True, must_be_file=True)

    # Determine output directory
    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_frames"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse resize
    resize_dims = None
    if resize:
        try:
            width, height = map(int, resize.lower().split("x"))
            resize_dims = (width, height)
        except ValueError:
            raise click.ClickException(f"Invalid resize format: {resize}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise click.ClickException(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_to_extract = total_frames // sample_rate
    if max_frames:
        frames_to_extract = min(frames_to_extract, max_frames)

    click.echo(f"Video: {video_path.name}")
    click.echo(f"Total frames: {total_frames}, FPS: {fps:.1f}")
    click.echo(f"Extracting {frames_to_extract} frames to {output_path}")

    frame_idx = 0
    extracted = 0

    with ProgressBar(total=frames_to_extract, description="Extracting") as bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                if max_frames and extracted >= max_frames:
                    break

                # Resize if requested
                if resize_dims:
                    frame = cv2.resize(frame, resize_dims)

                # Save frame
                frame_name = f"frame_{extracted:06d}.{format}"
                frame_path = output_path / frame_name
                cv2.imwrite(str(frame_path), frame)

                extracted += 1
                bar.update(1)

            frame_idx += 1

    cap.release()

    click.echo(f"\nExtracted {extracted} frames to {output_path}")


@utils.command("prepare-dataset")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "-o", "--output-dir",
    type=click.Path(),
    required=True,
    help="Output directory"
)
@click.option(
    "--split",
    type=float,
    default=None,
    help="Train/test split ratio (e.g., 0.8)"
)
@click.option(
    "--resize",
    type=str,
    default=None,
    help="Resize images to WxH"
)
@click.option(
    "--format",
    type=click.Choice(["png", "jpg"]),
    default=None,
    help="Convert images to format"
)
@click.pass_context
def prepare_dataset(
    ctx: click.Context,
    input_dir: str,
    output_dir: str,
    split: Optional[float],
    resize: Optional[str],
    format: Optional[str],
) -> None:
    """
    Prepare dataset for evaluation.

    INPUT_DIR: Directory with images

    Creates standardized directory structure for evaluation.
    """
    import shutil
    import random
    import cv2

    input_path = validate_path(input_dir, must_exist=True, must_be_dir=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse resize
    resize_dims = None
    if resize:
        try:
            width, height = map(int, resize.lower().split("x"))
            resize_dims = (width, height)
        except ValueError:
            raise click.ClickException(f"Invalid resize format: {resize}")

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in input_path.rglob("*")
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise click.ClickException(f"No images found in {input_path}")

    click.echo(f"Found {len(image_files)} images")

    # Split if requested
    if split:
        random.shuffle(image_files)
        split_idx = int(len(image_files) * split)
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]

        train_dir = output_path / "train"
        test_dir = output_path / "test"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        file_groups = [
            (train_files, train_dir, "train"),
            (test_files, test_dir, "test"),
        ]
    else:
        file_groups = [(image_files, output_path, "all")]

    # Process images
    for files, dest_dir, group_name in file_groups:
        click.echo(f"\nProcessing {group_name}: {len(files)} images")

        with ProgressBar(total=len(files), description=f"  {group_name}") as bar:
            for i, src_file in enumerate(files):
                # Determine output path
                if format:
                    out_name = f"image_{i:06d}.{format}"
                else:
                    out_name = f"image_{i:06d}{src_file.suffix}"
                out_file = dest_dir / out_name

                # Copy or process
                if resize_dims or format:
                    img = cv2.imread(str(src_file))
                    if resize_dims:
                        img = cv2.resize(img, resize_dims)
                    cv2.imwrite(str(out_file), img)
                else:
                    shutil.copy2(src_file, out_file)

                bar.update(1)

    click.echo(f"\nDataset prepared in {output_path}")


@utils.command("check-models")
@click.option(
    "--download",
    is_flag=True,
    help="Attempt to download missing models"
)
@click.pass_context
def check_models(ctx: click.Context, download: bool) -> None:
    """Check availability of required models."""
    click.echo("\nModel Availability Check")
    click.echo("=" * 40)

    # Define expected models
    models = [
        {
            "name": "InsightFace ArcFace",
            "type": "identity",
            "check": _check_insightface,
        },
        {
            "name": "LPIPS (AlexNet)",
            "type": "quality",
            "check": _check_lpips,
        },
    ]

    all_available = True

    for model in models:
        name = model["name"]
        try:
            available = model["check"]()
            status = "Available" if available else "Not found"
            symbol = "[+]" if available else "[-]"

            if not available:
                all_available = False

        except Exception as e:
            status = f"Error: {e}"
            symbol = "[!]"
            all_available = False

        click.echo(f"\n{symbol} {name}")
        click.echo(f"    Type: {model['type']}")
        click.echo(f"    Status: {status}")

    click.echo("\n" + "-" * 40)
    if all_available:
        click.echo("All models available.")
    else:
        click.echo("Some models are missing.")
        if not download:
            click.echo("Use --download to attempt automatic download.")


@utils.command("clear-cache")
@click.option(
    "--embeddings",
    is_flag=True,
    help="Clear embedding cache"
)
@click.option(
    "--all",
    "clear_all",
    is_flag=True,
    help="Clear all caches"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted"
)
@click.pass_context
def clear_cache(
    ctx: click.Context,
    embeddings: bool,
    clear_all: bool,
    dry_run: bool,
) -> None:
    """Clear cached data."""
    import shutil

    # Default cache locations
    cache_dirs = []

    home = Path.home()
    xdeid_cache = home / ".cache" / "xdeid3d"

    if clear_all or embeddings:
        cache_dirs.append(("embeddings", xdeid_cache / "embeddings"))

    if clear_all:
        cache_dirs.append(("models", xdeid_cache / "models"))
        cache_dirs.append(("temp", xdeid_cache / "temp"))

    if not cache_dirs:
        click.echo("No cache type specified. Use --embeddings or --all")
        return

    total_size = 0
    total_files = 0

    for name, cache_dir in cache_dirs:
        if cache_dir.exists():
            # Calculate size
            size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            files = sum(1 for f in cache_dir.rglob("*") if f.is_file())

            click.echo(f"\n{name}: {cache_dir}")
            click.echo(f"  Files: {files}")
            click.echo(f"  Size: {format_size(size)}")

            total_size += size
            total_files += files

            if not dry_run:
                shutil.rmtree(cache_dir)
                click.echo(f"  Deleted.")
        else:
            click.echo(f"\n{name}: {cache_dir}")
            click.echo(f"  (not found)")

    click.echo(f"\nTotal: {total_files} files, {format_size(total_size)}")

    if dry_run:
        click.echo("\n(dry run - nothing deleted)")


def _create_sphere_mesh(resolution: int = 50):
    """Create a simple sphere mesh."""
    import numpy as np

    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution * 2)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.cos(phi)
    z = np.sin(phi) * np.sin(theta)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)

    faces = []
    for i in range(resolution * 2 - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])

    return vertices, np.array(faces, dtype=np.int32)


def _check_insightface() -> bool:
    """Check if InsightFace is available."""
    try:
        import insightface
        return True
    except ImportError:
        return False


def _check_lpips() -> bool:
    """Check if LPIPS is available."""
    try:
        import lpips
        return True
    except ImportError:
        return False
