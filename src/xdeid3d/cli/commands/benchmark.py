"""
Benchmark commands for X-DeID3D CLI.

Commands for running performance benchmarks, comparing anonymizers,
and measuring inference speed.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np

from xdeid3d.cli.utils import (
    validate_path,
    get_device,
    ProgressBar,
    format_duration,
    create_output_dir,
)


@click.group()
@click.pass_context
def benchmark(ctx: click.Context) -> None:
    """Run performance benchmarks.

    \b
    Commands:
      anonymizer    Benchmark anonymizer performance
      metric        Benchmark metric computation speed
      throughput    Measure end-to-end throughput
      compare       Compare multiple anonymizers
      profile       Profile anonymizer execution
    """
    pass


@benchmark.command("anonymizer")
@click.argument("anonymizer_name", type=str)
@click.option(
    "--input",
    type=click.Path(exists=True),
    default=None,
    help="Input image or directory for benchmarking"
)
@click.option(
    "--resolution",
    type=str,
    default="512x512",
    help="Image resolution WxH (default: 512x512)"
)
@click.option(
    "--iterations", "-n",
    type=int,
    default=100,
    help="Number of iterations (default: 100)"
)
@click.option(
    "--warmup",
    type=int,
    default=10,
    help="Warmup iterations (default: 10)"
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size (default: 1)"
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (cpu, cuda)"
)
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output JSON file for results"
)
@click.pass_context
def benchmark_anonymizer(
    ctx: click.Context,
    anonymizer_name: str,
    input: Optional[str],
    resolution: str,
    iterations: int,
    warmup: int,
    batch_size: int,
    device: Optional[str],
    output: Optional[str],
) -> None:
    """
    Benchmark anonymizer performance.

    ANONYMIZER_NAME: Name of the registered anonymizer to benchmark

    Measures inference time, throughput, and memory usage.
    """
    import json

    from xdeid3d.anonymizers import get_anonymizer

    # Get device
    if device is None:
        device = ctx.obj.get("device")
    device = get_device(device)

    # Parse resolution
    try:
        width, height = map(int, resolution.lower().split("x"))
    except ValueError:
        raise click.ClickException(f"Invalid resolution format: {resolution}")

    click.echo(f"Benchmarking anonymizer: {anonymizer_name}")
    click.echo(f"Resolution: {width}x{height}")
    click.echo(f"Device: {device}")
    click.echo(f"Iterations: {iterations} (warmup: {warmup})")

    # Load anonymizer
    click.echo("\nLoading anonymizer...")
    try:
        anonymizer = get_anonymizer(anonymizer_name, device=device)
    except Exception as e:
        raise click.ClickException(f"Could not load anonymizer: {e}")

    # Prepare input
    if input:
        import cv2
        img = cv2.imread(input)
        if img is None:
            raise click.ClickException(f"Could not load image: {input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))
    else:
        # Generate random test image
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Prepare batch if needed
    if batch_size > 1:
        batch = np.stack([img] * batch_size)
    else:
        batch = img

    # Warmup
    click.echo(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = anonymizer.anonymize(batch)

    # Benchmark
    click.echo(f"Running benchmark ({iterations} iterations)...")

    times = []
    memory_samples = []

    # Try to get CUDA memory tracking
    cuda_available = False
    try:
        import torch
        if torch.cuda.is_available() and "cuda" in device:
            cuda_available = True
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    with ProgressBar(total=iterations, description="Benchmarking") as bar:
        for i in range(iterations):
            start = time.perf_counter()
            _ = anonymizer.anonymize(batch)
            end = time.perf_counter()

            times.append(end - start)

            # Sample memory periodically
            if cuda_available and i % 10 == 0:
                import torch
                memory_samples.append(torch.cuda.memory_allocated() / 1e6)

            bar.update(1)

    # Compute statistics
    times = np.array(times)
    total_images = iterations * batch_size

    results = {
        "anonymizer": anonymizer_name,
        "device": device,
        "resolution": f"{width}x{height}",
        "batch_size": batch_size,
        "iterations": iterations,
        "timing": {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
            "median_ms": float(np.median(times) * 1000),
            "p95_ms": float(np.percentile(times, 95) * 1000),
            "p99_ms": float(np.percentile(times, 99) * 1000),
        },
        "throughput": {
            "images_per_second": float(total_images / np.sum(times)),
            "batches_per_second": float(iterations / np.sum(times)),
        },
    }

    if memory_samples:
        results["memory_mb"] = {
            "mean": float(np.mean(memory_samples)),
            "max": float(np.max(memory_samples)),
        }
        if cuda_available:
            import torch
            results["memory_mb"]["peak"] = torch.cuda.max_memory_allocated() / 1e6

    # Print results
    click.echo("\n" + "=" * 50)
    click.echo("Benchmark Results")
    click.echo("=" * 50)

    click.echo(f"\nTiming (per batch):")
    click.echo(f"  Mean:   {results['timing']['mean_ms']:.2f} ms")
    click.echo(f"  Std:    {results['timing']['std_ms']:.2f} ms")
    click.echo(f"  Min:    {results['timing']['min_ms']:.2f} ms")
    click.echo(f"  Max:    {results['timing']['max_ms']:.2f} ms")
    click.echo(f"  P95:    {results['timing']['p95_ms']:.2f} ms")
    click.echo(f"  P99:    {results['timing']['p99_ms']:.2f} ms")

    click.echo(f"\nThroughput:")
    click.echo(f"  {results['throughput']['images_per_second']:.1f} images/sec")
    click.echo(f"  {results['throughput']['batches_per_second']:.1f} batches/sec")

    if "memory_mb" in results:
        click.echo(f"\nMemory (GPU):")
        click.echo(f"  Mean:   {results['memory_mb']['mean']:.1f} MB")
        click.echo(f"  Peak:   {results['memory_mb'].get('peak', results['memory_mb']['max']):.1f} MB")

    # Save results
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to: {output}")


@benchmark.command("metric")
@click.argument("metric_name", type=str)
@click.option(
    "--resolution",
    type=str,
    default="512x512",
    help="Image resolution WxH (default: 512x512)"
)
@click.option(
    "--iterations", "-n",
    type=int,
    default=100,
    help="Number of iterations"
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
@click.pass_context
def benchmark_metric(
    ctx: click.Context,
    metric_name: str,
    resolution: str,
    iterations: int,
    device: Optional[str],
    output: Optional[str],
) -> None:
    """
    Benchmark metric computation speed.

    METRIC_NAME: Name of the metric to benchmark
    """
    import json

    from xdeid3d.metrics import get_metric

    # Get device
    if device is None:
        device = ctx.obj.get("device")
    device = get_device(device)

    # Parse resolution
    try:
        width, height = map(int, resolution.lower().split("x"))
    except ValueError:
        raise click.ClickException(f"Invalid resolution format: {resolution}")

    click.echo(f"Benchmarking metric: {metric_name}")
    click.echo(f"Resolution: {width}x{height}")
    click.echo(f"Device: {device}")

    # Load metric
    try:
        metric = get_metric(metric_name, device=device)
    except Exception as e:
        raise click.ClickException(f"Could not load metric: {e}")

    # Generate test images
    img1 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        _ = metric.compute(img1, img2)

    # Benchmark
    times = []
    with ProgressBar(total=iterations, description="Benchmarking") as bar:
        for _ in range(iterations):
            start = time.perf_counter()
            _ = metric.compute(img1, img2)
            end = time.perf_counter()
            times.append(end - start)
            bar.update(1)

    times = np.array(times)

    results = {
        "metric": metric_name,
        "device": device,
        "resolution": f"{width}x{height}",
        "iterations": iterations,
        "timing": {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
        },
        "throughput": {
            "pairs_per_second": float(iterations / np.sum(times)),
        },
    }

    # Print results
    click.echo("\n" + "=" * 50)
    click.echo("Metric Benchmark Results")
    click.echo("=" * 50)
    click.echo(f"\nTiming:")
    click.echo(f"  Mean: {results['timing']['mean_ms']:.2f} ms")
    click.echo(f"  Std:  {results['timing']['std_ms']:.2f} ms")
    click.echo(f"\nThroughput: {results['throughput']['pairs_per_second']:.1f} pairs/sec")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to: {output}")


@benchmark.command("throughput")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "--anonymizer",
    type=str,
    required=True,
    help="Anonymizer to use"
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
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output JSON file"
)
@click.pass_context
def benchmark_throughput(
    ctx: click.Context,
    input_dir: str,
    anonymizer: str,
    metrics: str,
    device: Optional[str],
    output: Optional[str],
) -> None:
    """
    Measure end-to-end throughput.

    INPUT_DIR: Directory with input images

    Measures full pipeline throughput including loading,
    anonymization, and metric computation.
    """
    import json
    import cv2

    from xdeid3d.anonymizers import get_anonymizer
    from xdeid3d.metrics import get_metric
    from xdeid3d.cli.utils import parse_metrics

    input_path = validate_path(input_dir, must_exist=True, must_be_dir=True)

    # Get device
    if device is None:
        device = ctx.obj.get("device")
    device = get_device(device)

    # Find images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise click.ClickException(f"No images found in {input_path}")

    click.echo(f"Found {len(image_files)} images")
    click.echo(f"Anonymizer: {anonymizer}")
    click.echo(f"Device: {device}")

    # Load components
    anon = get_anonymizer(anonymizer, device=device)
    metric_names = parse_metrics(metrics)
    metric_objects = {name: get_metric(name, device=device) for name in metric_names}

    # Benchmark
    timings = {
        "load": [],
        "anonymize": [],
        "metrics": [],
        "total": [],
    }

    with ProgressBar(total=len(image_files), description="Processing") as bar:
        for img_file in image_files:
            total_start = time.perf_counter()

            # Load
            load_start = time.perf_counter()
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            load_end = time.perf_counter()
            timings["load"].append(load_end - load_start)

            # Anonymize
            anon_start = time.perf_counter()
            anon_img = anon.anonymize(img)
            anon_end = time.perf_counter()
            timings["anonymize"].append(anon_end - anon_start)

            # Compute metrics
            metric_start = time.perf_counter()
            for metric in metric_objects.values():
                _ = metric.compute(img, anon_img)
            metric_end = time.perf_counter()
            timings["metrics"].append(metric_end - metric_start)

            total_end = time.perf_counter()
            timings["total"].append(total_end - total_start)

            bar.update(1)

    # Compute statistics
    results = {
        "input_dir": str(input_path),
        "num_images": len(image_files),
        "anonymizer": anonymizer,
        "metrics": metric_names,
        "device": device,
    }

    for stage, times in timings.items():
        times = np.array(times)
        results[f"{stage}_timing"] = {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "total_s": float(np.sum(times)),
        }

    results["throughput"] = {
        "images_per_second": float(len(image_files) / np.sum(timings["total"])),
        "total_time_s": float(np.sum(timings["total"])),
    }

    # Print results
    click.echo("\n" + "=" * 50)
    click.echo("Throughput Benchmark Results")
    click.echo("=" * 50)

    for stage in ["load", "anonymize", "metrics", "total"]:
        click.echo(f"\n{stage.capitalize()}:")
        click.echo(f"  Mean: {results[f'{stage}_timing']['mean_ms']:.2f} ms")
        click.echo(f"  Total: {results[f'{stage}_timing']['total_s']:.2f} s")

    click.echo(f"\nOverall Throughput:")
    click.echo(f"  {results['throughput']['images_per_second']:.1f} images/sec")
    click.echo(f"  Total time: {format_duration(results['throughput']['total_time_s'])}")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to: {output}")


@benchmark.command("compare")
@click.argument("anonymizers", type=str, nargs=-1)
@click.option(
    "--resolution",
    type=str,
    default="512x512",
    help="Image resolution"
)
@click.option(
    "--iterations", "-n",
    type=int,
    default=50,
    help="Iterations per anonymizer"
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
@click.pass_context
def benchmark_compare(
    ctx: click.Context,
    anonymizers: Tuple[str, ...],
    resolution: str,
    iterations: int,
    device: Optional[str],
    output: Optional[str],
) -> None:
    """
    Compare performance of multiple anonymizers.

    ANONYMIZERS: Names of anonymizers to compare
    """
    import json

    from xdeid3d.anonymizers import get_anonymizer

    if len(anonymizers) < 2:
        raise click.ClickException("At least 2 anonymizers required for comparison")

    # Get device
    if device is None:
        device = ctx.obj.get("device")
    device = get_device(device)

    # Parse resolution
    try:
        width, height = map(int, resolution.lower().split("x"))
    except ValueError:
        raise click.ClickException(f"Invalid resolution format: {resolution}")

    click.echo(f"Comparing {len(anonymizers)} anonymizers")
    click.echo(f"Resolution: {width}x{height}")
    click.echo(f"Iterations: {iterations}")

    # Generate test image
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Benchmark each anonymizer
    results = {
        "resolution": f"{width}x{height}",
        "iterations": iterations,
        "device": device,
        "anonymizers": {},
    }

    for anon_name in anonymizers:
        click.echo(f"\nBenchmarking: {anon_name}")

        try:
            anon = get_anonymizer(anon_name, device=device)
        except Exception as e:
            click.echo(f"  Error loading: {e}", err=True)
            results["anonymizers"][anon_name] = {"error": str(e)}
            continue

        # Warmup
        for _ in range(5):
            _ = anon.anonymize(img)

        # Benchmark
        times = []
        with ProgressBar(total=iterations, description=f"  {anon_name}") as bar:
            for _ in range(iterations):
                start = time.perf_counter()
                _ = anon.anonymize(img)
                end = time.perf_counter()
                times.append(end - start)
                bar.update(1)

        times = np.array(times)
        results["anonymizers"][anon_name] = {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
            "images_per_second": float(iterations / np.sum(times)),
        }

    # Print comparison table
    click.echo("\n" + "=" * 60)
    click.echo("Comparison Results")
    click.echo("=" * 60)

    # Header
    click.echo(f"\n{'Anonymizer':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Img/sec':<12}")
    click.echo("-" * 56)

    # Sort by mean time
    sorted_anons = sorted(
        [(name, data) for name, data in results["anonymizers"].items() if "error" not in data],
        key=lambda x: x[1]["mean_ms"]
    )

    for name, data in sorted_anons:
        click.echo(
            f"{name:<20} {data['mean_ms']:<12.2f} {data['std_ms']:<12.2f} "
            f"{data['images_per_second']:<12.1f}"
        )

    # Show errors
    errors = [(name, data) for name, data in results["anonymizers"].items() if "error" in data]
    if errors:
        click.echo("\nFailed:")
        for name, data in errors:
            click.echo(f"  {name}: {data['error']}")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to: {output}")


@benchmark.command("profile")
@click.argument("anonymizer_name", type=str)
@click.option(
    "--resolution",
    type=str,
    default="512x512",
    help="Image resolution"
)
@click.option(
    "--iterations", "-n",
    type=int,
    default=20,
    help="Number of iterations to profile"
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
    help="Output file for profiling results"
)
@click.option(
    "--format",
    type=click.Choice(["text", "json", "flamegraph"]),
    default="text",
    help="Output format"
)
@click.pass_context
def benchmark_profile(
    ctx: click.Context,
    anonymizer_name: str,
    resolution: str,
    iterations: int,
    device: Optional[str],
    output: Optional[str],
    format: str,
) -> None:
    """
    Profile anonymizer execution.

    ANONYMIZER_NAME: Name of anonymizer to profile

    Provides detailed breakdown of execution time by component.
    """
    import cProfile
    import pstats
    import io
    import json

    from xdeid3d.anonymizers import get_anonymizer

    # Get device
    if device is None:
        device = ctx.obj.get("device")
    device = get_device(device)

    # Parse resolution
    try:
        width, height = map(int, resolution.lower().split("x"))
    except ValueError:
        raise click.ClickException(f"Invalid resolution format: {resolution}")

    click.echo(f"Profiling: {anonymizer_name}")
    click.echo(f"Resolution: {width}x{height}")
    click.echo(f"Iterations: {iterations}")

    # Load anonymizer
    anon = get_anonymizer(anonymizer_name, device=device)

    # Generate test image
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = anon.anonymize(img)

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(iterations):
        _ = anon.anonymize(img)

    profiler.disable()

    # Generate output
    if format == "text":
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(30)

        profile_text = stream.getvalue()

        if output:
            with open(output, "w") as f:
                f.write(profile_text)
            click.echo(f"Profile saved to: {output}")
        else:
            click.echo("\n" + profile_text)

    elif format == "json":
        stats = pstats.Stats(profiler)

        profile_data = {
            "anonymizer": anonymizer_name,
            "iterations": iterations,
            "resolution": f"{width}x{height}",
            "total_time": stats.total_tt,
            "functions": [],
        }

        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, name = func
            profile_data["functions"].append({
                "name": name,
                "filename": filename,
                "line": line,
                "calls": nc,
                "total_time": tt,
                "cumulative_time": ct,
            })

        # Sort by cumulative time
        profile_data["functions"].sort(key=lambda x: -x["cumulative_time"])
        profile_data["functions"] = profile_data["functions"][:50]

        if output:
            with open(output, "w") as f:
                json.dump(profile_data, f, indent=2)
            click.echo(f"Profile saved to: {output}")
        else:
            click.echo(json.dumps(profile_data, indent=2))

    elif format == "flamegraph":
        # Output in flamegraph format (folded stacks)
        stats = pstats.Stats(profiler)

        lines = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, name = func
            lines.append(f"{filename}:{name} {int(tt * 1e6)}")

        flamegraph_output = "\n".join(lines)

        if output:
            with open(output, "w") as f:
                f.write(flamegraph_output)
            click.echo(f"Flamegraph data saved to: {output}")
            click.echo("Use 'flamegraph.pl' to generate SVG")
        else:
            click.echo(flamegraph_output)
