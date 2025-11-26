"""
Generate commands for X-DeID3D CLI.

Commands for generating visualizations, heatmaps, and meshes
from evaluation results.
"""

from pathlib import Path
from typing import Optional

import click

from xdeid3d.cli.utils import (
    validate_path,
    get_device,
    ProgressBar,
    create_output_dir,
)


@click.group()
@click.pass_context
def generate(ctx: click.Context) -> None:
    """Generate visualizations and outputs from evaluation results."""
    pass


@generate.command("heatmap")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output file path (default: input_heatmap.png)"
)
@click.option(
    "--colormap", "-c",
    type=str,
    default="magma",
    help="Colormap name (default: magma)"
)
@click.option(
    "--resolution", "-r",
    type=int,
    default=72,
    help="Grid resolution (default: 72)"
)
@click.option(
    "--projection",
    type=click.Choice(["rectangular", "polar", "mollweide"]),
    default="rectangular",
    help="Projection type (default: rectangular)"
)
@click.option(
    "--metric",
    type=str,
    default=None,
    help="Metric name to visualize (default: first available)"
)
@click.pass_context
def generate_heatmap(
    ctx: click.Context,
    input_file: str,
    output: Optional[str],
    colormap: str,
    resolution: int,
    projection: str,
    metric: Optional[str],
) -> None:
    """
    Generate spherical heatmap from evaluation results.

    INPUT_FILE: Path to evaluation results (JSON or NPZ)
    """
    import json
    import numpy as np

    from xdeid3d.visualization import (
        HeatmapGenerator,
        create_2d_heatmap,
        create_polar_heatmap,
        create_mollweide_projection,
        save_figure,
    )

    input_path = validate_path(input_file, must_exist=True, must_be_file=True)

    # Determine output path
    if output is None:
        output = str(input_path.with_suffix('')) + "_heatmap.png"
    output_path = Path(output)

    click.echo(f"Loading evaluation results from {input_path}")

    # Load data
    if input_path.suffix == ".npz":
        data = np.load(input_path)
        yaws = data.get('yaws', data.get('yaw_grid', None))
        pitches = data.get('pitches', data.get('pitch_grid', None))
        scores = data.get('scores', data.get('score_grid', None))

        if scores is None:
            raise click.ClickException("Could not find score data in NPZ file")

    elif input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)

        # Extract angle-score pairs from JSON
        if 'frame_metrics' in data:
            frame_data = data['frame_metrics']
        elif isinstance(data, list):
            frame_data = data
        else:
            raise click.ClickException("Unknown JSON format")

        # Find metric to use
        if metric is None:
            # Find first numeric metric
            for key in frame_data[0].keys():
                if key not in ('yaw', 'pitch', 'frame_idx', 'frame_index'):
                    if isinstance(frame_data[0][key], (int, float)):
                        metric = key
                        break

        if metric is None:
            raise click.ClickException("No numeric metric found in data")

        click.echo(f"Using metric: {metric}")

        yaws = [d.get('yaw', 0) for d in frame_data]
        pitches = [d.get('pitch', np.pi/2) for d in frame_data]
        scores = [d.get(metric, 0) for d in frame_data]

    else:
        raise click.ClickException(f"Unsupported file format: {input_path.suffix}")

    # Generate heatmap
    click.echo(f"Generating heatmap with resolution {resolution}...")

    generator = HeatmapGenerator(resolution=resolution)
    generator.set_metric_name(metric or "score")

    for yaw, pitch, score in zip(yaws, pitches, scores):
        generator.add_score(float(yaw), float(pitch), float(score))

    heatmap = generator.generate()

    # Create visualization based on projection
    click.echo(f"Creating {projection} projection...")

    if projection == "rectangular":
        img = create_2d_heatmap(heatmap, colormap=colormap)
    elif projection == "polar":
        img = create_polar_heatmap(heatmap, colormap=colormap)
    elif projection == "mollweide":
        img = create_mollweide_projection(heatmap, colormap=colormap)

    # Save
    save_figure(img, output_path)
    click.echo(f"Saved heatmap to {output_path}")


@generate.command("mesh")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output PLY file path (default: input_colored.ply)"
)
@click.option(
    "--colormap", "-c",
    type=str,
    default="magma",
    help="Colormap name (default: magma)"
)
@click.option(
    "--bandwidth",
    type=float,
    default=0.5,
    help="Kernel bandwidth for interpolation (default: 0.5)"
)
@click.option(
    "--metric",
    type=str,
    default=None,
    help="Metric name to use for coloring"
)
@click.option(
    "--mesh-file",
    type=click.Path(exists=True),
    default=None,
    help="Base mesh PLY file to color"
)
@click.pass_context
def generate_mesh(
    ctx: click.Context,
    input_file: str,
    output: Optional[str],
    colormap: str,
    bandwidth: float,
    metric: Optional[str],
    mesh_file: Optional[str],
) -> None:
    """
    Generate colored mesh from evaluation results.

    INPUT_FILE: Path to evaluation results (JSON)
    """
    import json
    import numpy as np

    from xdeid3d.visualization import MeshExporter, read_ply

    input_path = validate_path(input_file, must_exist=True, must_be_file=True)

    # Load evaluation data
    click.echo(f"Loading evaluation results from {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    # Extract metrics
    if 'frame_metrics' in data:
        frame_data = data['frame_metrics']
    elif isinstance(data, list):
        frame_data = data
    else:
        raise click.ClickException("Unknown JSON format")

    # Create exporter
    exporter = MeshExporter(colormap=colormap, bandwidth=bandwidth)

    # Find metric
    if metric is None:
        for key in frame_data[0].keys():
            if key not in ('yaw', 'pitch', 'frame_idx', 'frame_index'):
                if isinstance(frame_data[0][key], (int, float)):
                    metric = key
                    break

    if metric is None:
        raise click.ClickException("No numeric metric found")

    click.echo(f"Using metric: {metric}")
    exporter.set_metric_name(metric)

    # Add scores
    for d in frame_data:
        if 'yaw' in d and 'pitch' in d:
            exporter.add_score(d['yaw'], d['pitch'], d.get(metric, 0))

    # Load or create base mesh
    if mesh_file:
        click.echo(f"Loading base mesh from {mesh_file}")
        vertices, faces, _, _ = read_ply(mesh_file)
    else:
        # Create simple sphere mesh
        click.echo("Creating sphere mesh (no base mesh provided)")
        vertices, faces = _create_sphere_mesh(resolution=50)

    # Determine output path
    if output is None:
        output = str(input_path.with_suffix('')) + "_colored.ply"

    # Export colored mesh
    click.echo("Generating colored mesh...")
    mesh = exporter.export_ply(output, vertices, faces)

    click.echo(f"Saved colored mesh to {output}")
    click.echo(f"  Vertices: {mesh.n_vertices}")
    click.echo(f"  Faces: {mesh.n_faces}")


@generate.command("figures")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory (default: input directory)"
)
@click.option(
    "--format",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    help="Output format (default: png)"
)
@click.option(
    "--dpi",
    type=int,
    default=150,
    help="Resolution in DPI (default: 150)"
)
@click.pass_context
def generate_figures(
    ctx: click.Context,
    input_file: str,
    output_dir: Optional[str],
    format: str,
    dpi: int,
) -> None:
    """
    Generate all figures from evaluation results.

    Creates time series plots, distributions, and summary figures.

    INPUT_FILE: Path to evaluation results (JSON)
    """
    import json
    import numpy as np

    from xdeid3d.visualization import (
        create_metric_plot,
        create_distribution_plot,
        create_summary_figure,
        save_figure,
    )

    input_path = validate_path(input_file, must_exist=True, must_be_file=True)

    # Determine output directory
    if output_dir is None:
        output_dir = input_path.parent / "figures"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    click.echo(f"Loading evaluation results from {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    # Extract frame metrics
    if 'frame_metrics' in data:
        frame_data = data['frame_metrics']
    elif isinstance(data, list):
        frame_data = data
    else:
        raise click.ClickException("Unknown JSON format")

    # Find numeric metrics
    metrics = {}
    for key in frame_data[0].keys():
        if key not in ('yaw', 'pitch', 'frame_idx', 'frame_index', 'timestamp'):
            values = [d.get(key) for d in frame_data]
            if all(isinstance(v, (int, float)) for v in values if v is not None):
                metrics[key] = [v if v is not None else 0 for v in values]

    click.echo(f"Found {len(metrics)} metrics to visualize")

    # Generate figures for each metric
    with ProgressBar(total=len(metrics) * 2, description="Generating figures") as bar:
        for metric_name, values in metrics.items():
            values_arr = np.array(values)

            # Time series
            ts_img = create_metric_plot(
                values_arr,
                title=f"{metric_name} Over Time",
                ylabel=metric_name,
                dpi=dpi,
            )
            ts_path = output_path / f"{metric_name}_timeseries.{format}"
            save_figure(ts_img, ts_path)
            bar.update(1)

            # Distribution
            dist_img = create_distribution_plot(
                values_arr,
                title=f"{metric_name} Distribution",
                xlabel=metric_name,
                dpi=dpi,
            )
            dist_path = output_path / f"{metric_name}_distribution.{format}"
            save_figure(dist_img, dist_path)
            bar.update(1)

    # Generate summary figure
    summary_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    summary_img = create_summary_figure(summary_metrics, title="Evaluation Summary", dpi=dpi)
    summary_path = output_path / f"summary.{format}"
    save_figure(summary_img, summary_path)

    click.echo(f"\nGenerated figures in {output_path}")
    click.echo(f"  - {len(metrics)} time series plots")
    click.echo(f"  - {len(metrics)} distribution plots")
    click.echo(f"  - 1 summary figure")


@generate.command("report")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output HTML file path"
)
@click.option(
    "--title",
    type=str,
    default="Evaluation Report",
    help="Report title"
)
@click.pass_context
def generate_report(
    ctx: click.Context,
    input_file: str,
    output: Optional[str],
    title: str,
) -> None:
    """
    Generate HTML report from evaluation results.

    INPUT_FILE: Path to evaluation results (JSON)
    """
    import json
    from datetime import datetime

    input_path = validate_path(input_file, must_exist=True, must_be_file=True)

    # Determine output path
    if output is None:
        output = str(input_path.with_suffix('.html'))

    # Load data
    click.echo(f"Loading evaluation results from {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    # Extract statistics
    if 'metric_stats' in data:
        stats = data['metric_stats']
    elif 'summary' in data:
        stats = data['summary']
    else:
        stats = {}

    # Generate HTML report
    html = _generate_html_report(title, data, stats)

    # Save
    Path(output).write_text(html)
    click.echo(f"Generated report: {output}")


def _create_sphere_mesh(resolution: int = 50) -> tuple:
    """Create a simple sphere mesh."""
    import numpy as np

    # Generate vertices
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution * 2)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.cos(phi)
    z = np.sin(phi) * np.sin(theta)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)

    # Generate faces
    faces = []
    for i in range(resolution * 2 - 1):
        for j in range(resolution - 1):
            idx = i * resolution + j
            faces.append([idx, idx + 1, idx + resolution])
            faces.append([idx + 1, idx + resolution + 1, idx + resolution])

    return vertices, np.array(faces, dtype=np.int32)


def _generate_html_report(title: str, data: dict, stats: dict) -> str:
    """Generate simple HTML report."""
    from datetime import datetime

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; }",
        "h1 { color: #333; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]

    # Add statistics table
    if stats:
        html_parts.append("<h2>Metric Statistics</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>")

        for metric, values in stats.items():
            if isinstance(values, dict):
                mean = values.get('mean', 'N/A')
                std = values.get('std', 'N/A')
                min_val = values.get('min', 'N/A')
                max_val = values.get('max', 'N/A')

                if isinstance(mean, float):
                    mean = f"{mean:.4f}"
                if isinstance(std, float):
                    std = f"{std:.4f}"
                if isinstance(min_val, float):
                    min_val = f"{min_val:.4f}"
                if isinstance(max_val, float):
                    max_val = f"{max_val:.4f}"

                html_parts.append(f"<tr><td>{metric}</td><td>{mean}</td><td>{std}</td><td>{min_val}</td><td>{max_val}</td></tr>")

        html_parts.append("</table>")

    # Add summary info
    html_parts.append("<h2>Summary</h2>")
    html_parts.append("<ul>")

    if 'total_samples' in data:
        html_parts.append(f"<li>Total Samples: {data['total_samples']}</li>")
    if 'successful_samples' in data:
        html_parts.append(f"<li>Successful: {data['successful_samples']}</li>")
    if 'failed_samples' in data:
        html_parts.append(f"<li>Failed: {data['failed_samples']}</li>")

    html_parts.append("</ul>")

    html_parts.extend([
        "</body>",
        "</html>",
    ])

    return "\n".join(html_parts)
