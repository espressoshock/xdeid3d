"""
X-DeID3D Command Line Interface.

Main entry point for the xdeid3d CLI application.
"""

import sys
from pathlib import Path
from typing import Optional

import click

from xdeid3d import __version__
from xdeid3d.cli.utils import (
    setup_logging,
    print_banner,
    validate_path,
    get_device,
)
from xdeid3d.cli.commands import generate as generate_commands
from xdeid3d.cli.commands import evaluate as evaluate_commands
from xdeid3d.cli.commands import benchmark as benchmark_commands

# Create main CLI group
@click.group()
@click.version_option(version=__version__, prog_name="xdeid3d")
@click.option(
    "-v", "--verbose",
    count=True,
    help="Increase verbosity (use -vv for debug output)"
)
@click.option(
    "-q", "--quiet",
    is_flag=True,
    help="Suppress non-error output"
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (cpu, cuda, cuda:0, etc.)"
)
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: bool, device: Optional[str]) -> None:
    """
    X-DeID3D: 3D Explainability Framework for Face Anonymization.

    A toolkit for evaluating and visualizing face anonymization
    performance across viewing angles.

    \b
    Commands:
      evaluate    Run evaluation on anonymized data
      generate    Generate synthetic data and visualizations
      benchmark   Run performance benchmarks
      config      Manage configuration
      info        Show system and package information

    Use 'xdeid3d COMMAND --help' for command-specific help.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Store global options
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["device"] = device

    # Setup logging based on verbosity
    if not quiet:
        log_level = "DEBUG" if verbose > 1 else "INFO" if verbose else "WARNING"
        setup_logging(log_level)

    # Print banner unless quiet
    if not quiet and verbose == 0:
        print_banner()


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show system and package information."""
    import platform

    click.echo("\nX-DeID3D System Information")
    click.echo("=" * 40)

    # Package info
    click.echo(f"X-DeID3D Version: {__version__}")
    click.echo(f"Python Version: {platform.python_version()}")
    click.echo(f"Platform: {platform.system()} {platform.release()}")

    # Check for optional dependencies
    click.echo("\nOptional Dependencies:")

    deps = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("insightface", "InsightFace"),
        ("lpips", "LPIPS"),
        ("trimesh", "Trimesh"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow"),
    ]

    for module, name in deps:
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "installed")
            click.echo(f"  {name}: {version}")
        except ImportError:
            click.echo(f"  {name}: not installed")

    # Device info
    click.echo("\nDevice Information:")
    try:
        import torch
        click.echo(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            click.echo(f"  CUDA Version: {torch.version.cuda}")
            click.echo(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                click.echo(f"  GPU {i}: {name} ({mem:.1f} GB)")
    except ImportError:
        click.echo("  PyTorch not installed")


@cli.command()
@click.option(
    "--format",
    type=click.Choice(["yaml", "json", "toml"]),
    default="yaml",
    help="Configuration format"
)
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output file path"
)
@click.pass_context
def init(ctx: click.Context, format: str, output: Optional[str]) -> None:
    """Initialize a new configuration file."""
    from xdeid3d.config import XDeID3DConfig

    # Create default config
    config = XDeID3DConfig()

    # Determine output path
    if output is None:
        output = f"xdeid3d_config.{format}"

    output_path = Path(output)

    # Export config
    if format == "yaml":
        try:
            import yaml
            content = yaml.dump(config.model_dump(), default_flow_style=False)
        except ImportError:
            click.echo("Error: PyYAML not installed. Use --format json instead.")
            sys.exit(1)
    elif format == "json":
        import json
        content = json.dumps(config.model_dump(), indent=2)
    elif format == "toml":
        try:
            import tomli_w
            content = tomli_w.dumps(config.model_dump())
        except ImportError:
            click.echo("Error: tomli_w not installed. Use --format json instead.")
            sys.exit(1)

    output_path.write_text(content)
    click.echo(f"Created configuration file: {output_path}")


@cli.group()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Manage configuration settings."""
    pass


@config.command("show")
@click.option(
    "-c", "--config-file",
    type=click.Path(exists=True),
    default=None,
    help="Configuration file path"
)
@click.pass_context
def config_show(ctx: click.Context, config_file: Optional[str]) -> None:
    """Show current configuration."""
    from xdeid3d.config import XDeID3DConfig

    if config_file:
        config = XDeID3DConfig.from_file(config_file)
    else:
        config = XDeID3DConfig()

    click.echo("\nCurrent Configuration:")
    click.echo("=" * 40)

    # Show as YAML-like format
    def show_dict(d, indent=0):
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                click.echo(f"{prefix}{key}:")
                show_dict(value, indent + 1)
            else:
                click.echo(f"{prefix}{key}: {value}")

    show_dict(config.model_dump())


@config.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
@click.pass_context
def config_validate(ctx: click.Context, config_file: str) -> None:
    """Validate a configuration file."""
    from xdeid3d.config import XDeID3DConfig

    try:
        config = XDeID3DConfig.from_file(config_file)
        click.echo(f"Configuration is valid: {config_file}")
    except Exception as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)


# Placeholder command groups for other commands
# These will be implemented in subsequent commits

# Add command groups from commands module
cli.add_command(evaluate_commands)
cli.add_command(generate_commands)
cli.add_command(benchmark_commands)


def main() -> int:
    """Main entry point for CLI."""
    try:
        cli(obj={})
        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
