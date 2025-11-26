"""
Pydantic configuration schemas for X-DeID3D.

This module defines all configuration classes using Pydantic v2 for
type-safe configuration management with validation and serialization.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import warnings

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "SynthesisConfig",
    "SamplingConfig",
    "MetricsConfig",
    "AnonymizerConfig",
    "EvaluationConfig",
    "VisualizationConfig",
    "OutputConfig",
    "Config",
    "EvaluationMode",
    "ColorMap",
]


class EvaluationMode(str, Enum):
    """Evaluation modes for the framework."""

    AGGREGATE = "aggregate"  # Population-level statistics
    INDIVIDUAL = "individual"  # Per-sample attribution
    BOTH = "both"  # Run both modes


class ColorMap(str, Enum):
    """Available colormaps for heatmap visualization."""

    VIRIDIS = "viridis"
    PLASMA = "plasma"
    MAGMA = "magma"
    INFERNO = "inferno"
    HOT = "hot"
    COOLWARM = "coolwarm"
    TURBO = "turbo"
    CIVIDIS = "cividis"


class SynthesisConfig(BaseModel):
    """Configuration for 3D head synthesis (SphereHead)."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path = Field(
        default=Path("models/spherehead-ckpt-025000.pkl"),
        description="Path to SphereHead model checkpoint",
    )
    truncation_psi: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Truncation parameter (0.5-0.8 = high quality, 1.0 = max diversity)",
    )
    truncation_cutoff: int = Field(
        default=14,
        ge=0,
        description="Number of layers to apply truncation",
    )
    neural_rendering_resolution: int = Field(
        default=128,
        ge=64,
        le=512,
        description="Resolution for neural rendering (higher = sharper but slower)",
    )
    depth_resolution: int = Field(
        default=48,
        ge=16,
        description="Base depth resolution for volume rendering",
    )
    sample_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=4.0,
        description="Multiplier for depth sampling (higher = fewer artifacts)",
    )
    white_background: bool = Field(
        default=True,
        description="Use white background instead of black",
    )
    cfg: Literal["Head", "FFHQ", "Cats"] = Field(
        default="Head",
        description="Model configuration preset",
    )

    @field_validator("model_path", mode="before")
    @classmethod
    def validate_model_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v


class SamplingConfig(BaseModel):
    """Configuration for spherical sampling."""

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(
        default=180,
        ge=10,
        le=1000,
        description="Number of viewing angles to sample",
    )
    angular_resolution_deg: float = Field(
        default=2.0,
        ge=0.5,
        le=30.0,
        description="Angular resolution between samples in degrees",
    )
    yaw_range: tuple[float, float] = Field(
        default=(0.0, 360.0),
        description="Yaw angle range in degrees (min, max)",
    )
    pitch_range: tuple[float, float] = Field(
        default=(30.0, 150.0),
        description="Pitch angle range in degrees (min, max)",
    )
    sampling_pattern: Literal["uniform", "fibonacci", "custom"] = Field(
        default="uniform",
        description="Spherical sampling pattern",
    )
    camera_radius: float = Field(
        default=2.7,
        ge=1.0,
        le=10.0,
        description="Distance from camera to subject",
    )

    @field_validator("yaw_range", "pitch_range", mode="before")
    @classmethod
    def validate_range(cls, v: Any) -> tuple[float, float]:
        """Validate angle ranges."""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (float(v[0]), float(v[1]))
        raise ValueError("Range must be a tuple of (min, max)")


class MetricsConfig(BaseModel):
    """Configuration for metric computation."""

    model_config = ConfigDict(extra="forbid")

    enabled_metrics: List[str] = Field(
        default=[
            "arcface_cosine_distance",
            "psnr",
            "ssim",
            "temporal_identity_consistency",
        ],
        description="List of metrics to compute",
    )
    primary_metric: str = Field(
        default="arcface_cosine_distance",
        description="Primary metric for optimization/ranking",
    )
    identity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for identity verification",
    )
    embedding_model: str = Field(
        default="buffalo_l",
        description="Face recognition model for embeddings (insightface model name)",
    )
    embedding_cache: bool = Field(
        default=True,
        description="Enable embedding caching for performance",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for metric computation",
    )

    @model_validator(mode="after")
    def validate_primary_in_enabled(self) -> "MetricsConfig":
        """Ensure primary metric is in enabled list."""
        if self.primary_metric not in self.enabled_metrics:
            self.enabled_metrics.append(self.primary_metric)
        return self


class AnonymizerConfig(BaseModel):
    """Configuration for face anonymizer."""

    model_config = ConfigDict(extra="allow")  # Allow custom config keys

    type: str = Field(
        default="blur",
        description="Anonymizer type (e.g., 'blur', 'pixelate', 'custom')",
    )
    name: Optional[str] = Field(
        default=None,
        description="Custom anonymizer name (for registry lookup)",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Anonymizer-specific configuration",
    )

    # Common anonymizer settings
    kernel_size: int = Field(
        default=51,
        ge=3,
        description="Blur kernel size (for blur-based anonymizers)",
    )
    sigma: float = Field(
        default=20.0,
        ge=0.0,
        description="Gaussian blur sigma",
    )
    block_size: int = Field(
        default=10,
        ge=2,
        description="Pixelation block size",
    )

    @model_validator(mode="after")
    def merge_config(self) -> "AnonymizerConfig":
        """Merge top-level settings into config dict."""
        # Allow users to specify settings at top level or in config
        for key in ["kernel_size", "sigma", "block_size"]:
            val = getattr(self, key, None)
            if val is not None and key not in self.config:
                self.config[key] = val
        return self


class EvaluationConfig(BaseModel):
    """Configuration for evaluation pipeline."""

    model_config = ConfigDict(extra="forbid")

    mode: EvaluationMode = Field(
        default=EvaluationMode.AGGREGATE,
        description="Evaluation mode (aggregate, individual, or both)",
    )
    seeds: List[int] = Field(
        default=[0],
        description="Random seeds for synthetic identities",
    )
    num_frames: int = Field(
        default=90,
        ge=10,
        le=1000,
        description="Number of frames per video",
    )
    fps: int = Field(
        default=30,
        ge=1,
        le=120,
        description="Video frame rate",
    )
    voxel_resolution: int = Field(
        default=128,
        ge=64,
        le=1024,
        description="3D voxel resolution for mesh extraction",
    )
    extract_mesh: bool = Field(
        default=True,
        description="Extract 3D mesh for heatmap visualization",
    )
    save_intermediate: bool = Field(
        default=False,
        description="Save intermediate results (frames, embeddings)",
    )

    @field_validator("seeds", mode="before")
    @classmethod
    def parse_seeds(cls, v: Any) -> List[int]:
        """Parse seed string like '0-5' or '0,1,2,3'."""
        if isinstance(v, str):
            if "-" in v:
                start, end = map(int, v.split("-"))
                return list(range(start, end + 1))
            return [int(s.strip()) for s in v.split(",")]
        return list(v) if isinstance(v, (list, tuple)) else [v]


class VisualizationConfig(BaseModel):
    """Configuration for visualization output."""

    model_config = ConfigDict(extra="forbid")

    colormap: ColorMap = Field(
        default=ColorMap.VIRIDIS,
        description="Colormap for heatmap visualization",
    )
    resolution: tuple[int, int] = Field(
        default=(512, 512),
        description="Output image resolution (width, height)",
    )
    generate_video: bool = Field(
        default=True,
        description="Generate video outputs",
    )
    generate_mesh: bool = Field(
        default=True,
        description="Generate 3D mesh outputs",
    )
    generate_plots: bool = Field(
        default=True,
        description="Generate analysis plots",
    )
    mesh_format: Literal["ply", "obj", "glb"] = Field(
        default="ply",
        description="3D mesh output format",
    )
    video_codec: str = Field(
        default="libx264",
        description="Video encoding codec",
    )

    @field_validator("resolution", mode="before")
    @classmethod
    def validate_resolution(cls, v: Any) -> tuple[int, int]:
        """Validate resolution tuple."""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        if isinstance(v, int):
            return (v, v)
        raise ValueError("Resolution must be (width, height) or single int")


class OutputConfig(BaseModel):
    """Configuration for output paths and formats."""

    model_config = ConfigDict(extra="forbid")

    base_dir: Path = Field(
        default=Path("results"),
        description="Base output directory",
    )
    experiment_name: Optional[str] = Field(
        default=None,
        description="Experiment name (auto-generated if None)",
    )
    format: Literal["json", "csv", "parquet"] = Field(
        default="json",
        description="Metrics output format",
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite existing results",
    )
    compress: bool = Field(
        default=False,
        description="Compress output files",
    )

    @field_validator("base_dir", mode="before")
    @classmethod
    def validate_base_dir(cls, v: Any) -> Path:
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v


class Config(BaseModel):
    """
    Main configuration class for X-DeID3D.

    This class combines all sub-configurations and provides methods
    for loading from TOML/YAML files and environment variables.

    Example:
        >>> config = Config.from_toml("xdeid3d.toml")
        >>> config = Config(
        ...     synthesis=SynthesisConfig(truncation_psi=0.5),
        ...     metrics=MetricsConfig(enabled_metrics=["psnr"])
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    synthesis: SynthesisConfig = Field(
        default_factory=SynthesisConfig,
        description="Synthesis configuration",
    )
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Sampling configuration",
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics configuration",
    )
    anonymizer: AnonymizerConfig = Field(
        default_factory=AnonymizerConfig,
        description="Anonymizer configuration",
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation configuration",
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization configuration",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration",
    )

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from TOML file.

        Args:
            path: Path to TOML configuration file

        Returns:
            Config instance
        """
        import sys

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Use tomli for Python < 3.11, tomllib for >= 3.11
        if sys.version_info >= (3, 11):
            import tomllib

            with open(path, "rb") as f:
                data = tomllib.load(f)
        else:
            import tomli

            with open(path, "rb") as f:
                data = tomli.load(f)

        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from file (auto-detect format).

        Args:
            path: Path to configuration file (.toml or .yaml/.yml)

        Returns:
            Config instance
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".toml":
            return cls.from_toml(path)
        elif suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

    def to_toml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to TOML file.

        Args:
            path: Output path
        """
        import tomli_w

        path = Path(path)
        data = self.model_dump(mode="json")

        # Convert Path objects to strings
        data = _convert_paths_to_strings(data)

        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output path
        """
        import yaml

        path = Path(path)
        data = self.model_dump(mode="json")
        data = _convert_paths_to_strings(data)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _convert_paths_to_strings(obj: Any) -> Any:
    """Recursively convert Path objects to strings for serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(v) for v in obj]
    elif isinstance(obj, tuple):
        return list(_convert_paths_to_strings(v) for v in obj)
    return obj
