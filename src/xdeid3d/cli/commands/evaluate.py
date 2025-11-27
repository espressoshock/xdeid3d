"""
Evaluate commands for X-DeID3D CLI.

Commands for running evaluation on anonymized data, including
single samples, video sequences, and batch processing.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import click

from xdeid3d.cli.utils import (
    validate_path,
    get_device,
    ProgressBar,
    create_output_dir,
    parse_metrics,
    parse_seeds,
)


@click.group()
@click.pass_context
def evaluate(ctx: click.Context) -> None:
    """Run evaluation on anonymized data.

    \b
    Commands:
      run       Run full evaluation pipeline
      single    Evaluate single image pair
      video     Evaluate video sequence
      compare   Compare multiple evaluation results
      metrics   List available metrics
    """
    pass


@evaluate.command("run")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output directory for results (default: input_dir/evaluation)"
)
@click.option(
    "-c", "--config",
    type=click.Path(exists=True),
    default=None,
    help="Configuration file path"
)
@click.option(
    "--metrics",
    type=str,
    default="standard",
    help="Metrics to compute (standard, full, or comma-separated list)"
)
@click.option(
    "--mode",
    type=click.Choice(["aggregate", "single", "both"]),
    default="both",
    help="Evaluation mode (default: both)"
)
@click.option(
    "--anonymizer",
    type=str,
    default=None,
    help="Anonymizer name used (for metadata)"
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (cpu, cuda, cuda:0)"
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size for processing (default: 1)"
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="Number of data loading workers (default: 4)"
)
@click.pass_context
def evaluate_run(
    ctx: click.Context,
    input_dir: str,
    output: Optional[str],
    config: Optional[str],
    metrics: str,
    mode: str,
    anonymizer: Optional[str],
    device: Optional[str],
    batch_size: int,
    workers: int,
) -> None:
    """
    Run full evaluation pipeline on a dataset.

    INPUT_DIR: Directory containing original and anonymized image pairs.
    Expected structure:
        input_dir/
            original/
                image001.png
                image002.png
            anonymized/
                image001.png
                image002.png

    Or for video evaluation:
        input_dir/
            original/
                frame_0001.png
                ...
            anonymized/
                frame_0001.png
                ...
            poses.json (optional)
    """
    import json
    import numpy as np

    from xdeid3d.config import XDeID3DConfig
    from xdeid3d.evaluation import EvaluationPipeline, EvaluationResult
    from xdeid3d.metrics import get_metric

    input_path = validate_path(input_dir, must_exist=True, must_be_dir=True)

    # Load or create config
    if config:
        cfg = XDeID3DConfig.from_file(config)
    else:
        cfg = XDeID3DConfig()

    # Override device if specified
    if device is None:
        device = ctx.obj.get("device")
    device = get_device(device)

    # Parse metrics
    metric_names = parse_metrics(metrics)

    click.echo(f"Running evaluation on: {input_path}")
    click.echo(f"Metrics: {', '.join(metric_names)}")
    click.echo(f"Mode: {mode}")
    click.echo(f"Device: {device}")

    # Determine output directory
    if output is None:
        output = input_path / "evaluation"
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find image pairs
    original_dir = input_path / "original"
    anonymized_dir = input_path / "anonymized"

    if not original_dir.exists() or not anonymized_dir.exists():
        # Try flat structure
        original_dir = input_path
        anonymized_dir = input_path

    # Collect image pairs
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    original_files = sorted([
        f for f in original_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    click.echo(f"Found {len(original_files)} images to evaluate")

    # Create pipeline
    pipeline = EvaluationPipeline(config=cfg)

    # Load metrics
    for metric_name in metric_names:
        try:
            metric = get_metric(metric_name, device=device)
            pipeline.add_metric(metric)
        except Exception as e:
            click.echo(f"Warning: Could not load metric '{metric_name}': {e}", err=True)

    # Load poses if available
    poses_file = input_path / "poses.json"
    poses_data = None
    if poses_file.exists():
        with open(poses_file) as f:
            poses_data = json.load(f)
        click.echo(f"Loaded pose data from {poses_file}")

    # Run evaluation
    results = []
    with ProgressBar(total=len(original_files), description="Evaluating") as bar:
        for i, orig_file in enumerate(original_files):
            # Find corresponding anonymized file
            anon_file = anonymized_dir / orig_file.name
            if not anon_file.exists():
                # Try without extension match
                anon_candidates = list(anonymized_dir.glob(f"{orig_file.stem}.*"))
                if anon_candidates:
                    anon_file = anon_candidates[0]
                else:
                    click.echo(f"Warning: No match for {orig_file.name}", err=True)
                    bar.update(1)
                    continue

            # Load images
            import cv2
            original_img = cv2.imread(str(orig_file))
            anonymized_img = cv2.imread(str(anon_file))

            if original_img is None or anonymized_img is None:
                click.echo(f"Warning: Could not load {orig_file.name}", err=True)
                bar.update(1)
                continue

            # Convert BGR to RGB
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            anonymized_img = cv2.cvtColor(anonymized_img, cv2.COLOR_BGR2RGB)

            # Get pose if available
            pose = None
            if poses_data:
                frame_key = orig_file.stem
                if frame_key in poses_data:
                    pose = poses_data[frame_key]
                elif i < len(poses_data.get("frames", [])):
                    pose = poses_data["frames"][i]

            # Evaluate
            result = pipeline.evaluate_pair(
                original_img,
                anonymized_img,
                pose=pose,
            )

            result_dict = {
                "frame_idx": i,
                "filename": orig_file.name,
            }

            if pose:
                result_dict["yaw"] = pose.get("yaw", 0)
                result_dict["pitch"] = pose.get("pitch", np.pi / 2)

            result_dict.update(result.scores)
            results.append(result_dict)

            bar.update(1)

    # Compute aggregate statistics
    click.echo("\nComputing statistics...")

    stats = {}
    for metric_name in metric_names:
        values = [r.get(metric_name) for r in results if metric_name in r]
        if values:
            values = np.array(values)
            stats[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

    # Save results
    output_data = {
        "config": {
            "input_dir": str(input_path),
            "metrics": metric_names,
            "mode": mode,
            "anonymizer": anonymizer,
        },
        "metric_stats": stats,
        "frame_metrics": results,
        "total_samples": len(original_files),
        "successful_samples": len(results),
        "failed_samples": len(original_files) - len(results),
    }

    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)

    click.echo(f"\nResults saved to: {results_file}")

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Evaluation Summary")
    click.echo("=" * 50)

    for metric_name, metric_stats in stats.items():
        click.echo(f"\n{metric_name}:")
        click.echo(f"  Mean:   {metric_stats['mean']:.4f}")
        click.echo(f"  Std:    {metric_stats['std']:.4f}")
        click.echo(f"  Min:    {metric_stats['min']:.4f}")
        click.echo(f"  Max:    {metric_stats['max']:.4f}")


@evaluate.command("single")
@click.argument("original", type=click.Path(exists=True))
@click.argument("anonymized", type=click.Path(exists=True))
@click.option(
    "--metrics",
    type=str,
    default="standard",
    help="Metrics to compute"
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use"
)
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output JSON file"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output"
)
@click.pass_context
def evaluate_single(
    ctx: click.Context,
    original: str,
    anonymized: str,
    metrics: str,
    device: Optional[str],
    output: Optional[str],
    verbose: bool,
) -> None:
    """
    Evaluate a single image pair.

    ORIGINAL: Path to original image
    ANONYMIZED: Path to anonymized image
    """
    import json
    import cv2

    from xdeid3d.metrics import get_metric

    original_path = validate_path(original, must_exist=True, must_be_file=True)
    anonymized_path = validate_path(anonymized, must_exist=True, must_be_file=True)

    # Get device
    if device is None:
        device = ctx.obj.get("device")
    device = get_device(device)

    # Parse metrics
    metric_names = parse_metrics(metrics)

    if verbose:
        click.echo(f"Original: {original_path}")
        click.echo(f"Anonymized: {anonymized_path}")
        click.echo(f"Metrics: {', '.join(metric_names)}")

    # Load images
    original_img = cv2.imread(str(original_path))
    anonymized_img = cv2.imread(str(anonymized_path))

    if original_img is None:
        raise click.ClickException(f"Could not load image: {original_path}")
    if anonymized_img is None:
        raise click.ClickException(f"Could not load image: {anonymized_path}")

    # Convert BGR to RGB
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    anonymized_img = cv2.cvtColor(anonymized_img, cv2.COLOR_BGR2RGB)

    # Compute metrics
    results = {}
    for metric_name in metric_names:
        try:
            metric = get_metric(metric_name, device=device)
            score = metric.compute(original_img, anonymized_img)
            results[metric_name] = float(score)

            if verbose:
                click.echo(f"  {metric_name}: {score:.4f}")

        except Exception as e:
            click.echo(f"Warning: {metric_name} failed: {e}", err=True)
            results[metric_name] = None

    # Output results
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to: {output}")
    else:
        click.echo("\nResults:")
        for name, value in results.items():
            if value is not None:
                click.echo(f"  {name}: {value:.4f}")
            else:
                click.echo(f"  {name}: failed")


@evaluate.command("video")
@click.argument("original_video", type=click.Path(exists=True))
@click.argument("anonymized_video", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output JSON file"
)
@click.option(
    "--metrics",
    type=str,
    default="standard",
    help="Metrics to compute"
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use"
)
@click.option(
    "--sample-rate",
    type=int,
    default=1,
    help="Sample every Nth frame (default: 1)"
)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Maximum frames to process"
)
@click.option(
    "--poses-file",
    type=click.Path(exists=True),
    default=None,
    help="JSON file with frame poses"
)
@click.pass_context
def evaluate_video(
    ctx: click.Context,
    original_video: str,
    anonymized_video: str,
    output: Optional[str],
    metrics: str,
    device: Optional[str],
    sample_rate: int,
    max_frames: Optional[int],
    poses_file: Optional[str],
) -> None:
    """
    Evaluate video sequence frame by frame.

    ORIGINAL_VIDEO: Path to original video file
    ANONYMIZED_VIDEO: Path to anonymized video file
    """
    import json
    import cv2
    import numpy as np

    from xdeid3d.metrics import get_metric

    original_path = validate_path(original_video, must_exist=True, must_be_file=True)
    anonymized_path = validate_path(anonymized_video, must_exist=True, must_be_file=True)

    # Get device
    if device is None:
        device = ctx.obj.get("device")
    device = get_device(device)

    # Parse metrics
    metric_names = parse_metrics(metrics)

    # Load poses if provided
    poses = None
    if poses_file:
        with open(poses_file) as f:
            poses = json.load(f)

    # Open videos
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_anon = cv2.VideoCapture(str(anonymized_path))

    if not cap_orig.isOpened():
        raise click.ClickException(f"Could not open video: {original_path}")
    if not cap_anon.isOpened():
        raise click.ClickException(f"Could not open video: {anonymized_path}")

    # Get video properties
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)

    if max_frames:
        total_frames = min(total_frames, max_frames * sample_rate)

    frames_to_process = total_frames // sample_rate

    click.echo(f"Video: {original_path.name}")
    click.echo(f"Total frames: {total_frames}, FPS: {fps:.1f}")
    click.echo(f"Processing {frames_to_process} frames (sample rate: {sample_rate})")

    # Load metrics
    metric_objects = {}
    for metric_name in metric_names:
        try:
            metric_objects[metric_name] = get_metric(metric_name, device=device)
        except Exception as e:
            click.echo(f"Warning: Could not load {metric_name}: {e}", err=True)

    # Process frames
    results = []
    frame_idx = 0

    with ProgressBar(total=frames_to_process, description="Processing") as bar:
        while True:
            ret_orig, frame_orig = cap_orig.read()
            ret_anon, frame_anon = cap_anon.read()

            if not ret_orig or not ret_anon:
                break

            # Skip frames based on sample rate
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            if max_frames and len(results) >= max_frames:
                break

            # Convert BGR to RGB
            frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
            frame_anon = cv2.cvtColor(frame_anon, cv2.COLOR_BGR2RGB)

            # Compute metrics
            result = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0,
            }

            # Add pose if available
            if poses:
                if isinstance(poses, list) and frame_idx < len(poses):
                    pose = poses[frame_idx]
                    result["yaw"] = pose.get("yaw", 0)
                    result["pitch"] = pose.get("pitch", np.pi / 2)
                elif "frames" in poses and frame_idx < len(poses["frames"]):
                    pose = poses["frames"][frame_idx]
                    result["yaw"] = pose.get("yaw", 0)
                    result["pitch"] = pose.get("pitch", np.pi / 2)

            for metric_name, metric in metric_objects.items():
                try:
                    score = metric.compute(frame_orig, frame_anon)
                    result[metric_name] = float(score)
                except Exception:
                    result[metric_name] = None

            results.append(result)
            frame_idx += 1
            bar.update(1)

    cap_orig.release()
    cap_anon.release()

    # Compute statistics
    stats = {}
    for metric_name in metric_names:
        values = [r.get(metric_name) for r in results if r.get(metric_name) is not None]
        if values:
            values = np.array(values)
            stats[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    # Output results
    output_data = {
        "video_info": {
            "original": str(original_path),
            "anonymized": str(anonymized_path),
            "total_frames": total_frames,
            "fps": fps,
            "sample_rate": sample_rate,
        },
        "metric_stats": stats,
        "frame_metrics": results,
    }

    if output:
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"\nResults saved to: {output}")

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Video Evaluation Summary")
    click.echo("=" * 50)

    for metric_name, metric_stats in stats.items():
        click.echo(f"\n{metric_name}:")
        click.echo(f"  Mean: {metric_stats['mean']:.4f}")
        click.echo(f"  Std:  {metric_stats['std']:.4f}")


@evaluate.command("compare")
@click.argument("results_files", type=click.Path(exists=True), nargs=-1)
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output comparison file"
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format (default: table)"
)
@click.pass_context
def evaluate_compare(
    ctx: click.Context,
    results_files: Tuple[str, ...],
    output: Optional[str],
    format: str,
) -> None:
    """
    Compare multiple evaluation results.

    RESULTS_FILES: Paths to evaluation result JSON files
    """
    import json

    if len(results_files) < 2:
        raise click.ClickException("At least 2 results files required for comparison")

    # Load all results
    all_results = []
    for path in results_files:
        with open(path) as f:
            data = json.load(f)
            data["_source"] = Path(path).stem
            all_results.append(data)

    # Extract metric stats
    comparison = {}
    all_metrics = set()

    for result in all_results:
        source = result["_source"]
        comparison[source] = {}

        if "metric_stats" in result:
            for metric, stats in result["metric_stats"].items():
                all_metrics.add(metric)
                comparison[source][metric] = stats.get("mean", 0)

    # Output comparison
    if format == "table":
        # Print comparison table
        click.echo("\nComparison Results")
        click.echo("=" * 60)

        # Header
        sources = list(comparison.keys())
        header = f"{'Metric':<25}"
        for source in sources:
            header += f"{source[:12]:>12}"
        click.echo(header)
        click.echo("-" * 60)

        # Rows
        for metric in sorted(all_metrics):
            row = f"{metric:<25}"
            for source in sources:
                value = comparison[source].get(metric, 0)
                row += f"{value:>12.4f}"
            click.echo(row)

    elif format == "json":
        comparison_data = {
            "sources": list(comparison.keys()),
            "metrics": {
                metric: {
                    source: comparison[source].get(metric)
                    for source in comparison
                }
                for metric in all_metrics
            }
        }

        if output:
            with open(output, "w") as f:
                json.dump(comparison_data, f, indent=2)
            click.echo(f"Comparison saved to: {output}")
        else:
            click.echo(json.dumps(comparison_data, indent=2))

    elif format == "csv":
        import csv
        from io import StringIO

        buffer = StringIO()
        writer = csv.writer(buffer)

        # Header
        sources = list(comparison.keys())
        writer.writerow(["metric"] + sources)

        # Rows
        for metric in sorted(all_metrics):
            row = [metric] + [comparison[s].get(metric, "") for s in sources]
            writer.writerow(row)

        if output:
            with open(output, "w") as f:
                f.write(buffer.getvalue())
            click.echo(f"Comparison saved to: {output}")
        else:
            click.echo(buffer.getvalue())


@evaluate.command("metrics")
@click.option(
    "--category",
    type=click.Choice(["all", "identity", "quality", "temporal", "explainability"]),
    default="all",
    help="Filter by category"
)
@click.pass_context
def evaluate_metrics(ctx: click.Context, category: str) -> None:
    """List available metrics."""
    from xdeid3d.metrics import list_metrics

    click.echo("\nAvailable Metrics")
    click.echo("=" * 50)

    metrics = list_metrics()

    for name, info in sorted(metrics.items()):
        metric_category = info.get("category", "unknown")

        if category != "all" and metric_category != category:
            continue

        click.echo(f"\n{name}")
        click.echo(f"  Category: {metric_category}")
        if "description" in info:
            click.echo(f"  Description: {info['description']}")
        if "range" in info:
            click.echo(f"  Range: {info['range']}")
