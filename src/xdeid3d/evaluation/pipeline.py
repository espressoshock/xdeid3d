"""
Evaluation pipeline orchestrator.

Main entry point for running complete evaluation workflows,
coordinating providers, modes, metrics, and output generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import json
import time
import uuid

from xdeid3d.evaluation.data import (
    EvaluationSample,
    EvaluationResult,
    AggregatedResults,
    ExperimentMetadata,
    EvaluationStatus,
)
from xdeid3d.evaluation.providers import SampleProvider
from xdeid3d.evaluation.modes.base import EvaluationMode, MetricSuite
from xdeid3d.evaluation.modes.single import SingleSampleMode
from xdeid3d.evaluation.modes.aggregate import AggregateMode
from xdeid3d.evaluation.modes.spherical import SphericalMode

__all__ = [
    "EvaluationPipeline",
    "PipelineConfig",
    "PipelineResult",
]


@dataclass
class PipelineConfig:
    """
    Configuration for evaluation pipeline.

    Args:
        mode: Evaluation mode ('single', 'aggregate', 'spherical')
        metrics: List of metric names or 'standard', 'full'
        output_dir: Directory for saving results
        save_per_sample: Save individual sample results
        save_aggregated: Save aggregated statistics
        generate_report: Generate HTML/JSON report
        verbose: Print progress information
    """
    mode: str = "aggregate"
    metrics: Union[str, List[str]] = "standard"
    output_dir: Optional[str] = None
    save_per_sample: bool = True
    save_aggregated: bool = True
    generate_report: bool = True
    verbose: bool = True

    # Mode-specific options
    fail_on_error: bool = False
    compute_confidence: bool = True
    detect_outliers: bool = True

    # Spherical mode options
    interpolation_bandwidth: float = 0.5
    grid_resolution: int = 72
    primary_metric: str = "arcface_cosine_distance"

    # Experiment metadata
    experiment_name: str = ""
    experiment_description: str = ""
    anonymizer_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode,
            'metrics': self.metrics,
            'output_dir': self.output_dir,
            'save_per_sample': self.save_per_sample,
            'save_aggregated': self.save_aggregated,
            'generate_report': self.generate_report,
            'verbose': self.verbose,
            'fail_on_error': self.fail_on_error,
            'compute_confidence': self.compute_confidence,
            'detect_outliers': self.detect_outliers,
            'interpolation_bandwidth': self.interpolation_bandwidth,
            'grid_resolution': self.grid_resolution,
            'primary_metric': self.primary_metric,
            'experiment_name': self.experiment_name,
            'experiment_description': self.experiment_description,
            'anonymizer_name': self.anonymizer_name,
        }


@dataclass
class PipelineResult:
    """
    Result of an evaluation pipeline run.

    Contains aggregated results, metadata, and output paths.
    """
    experiment_id: str
    metadata: ExperimentMetadata
    aggregated: AggregatedResults
    output_dir: Optional[Path] = None
    output_files: Dict[str, Path] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def summary(self) -> str:
        """Generate a text summary."""
        lines = [
            f"Pipeline Result: {self.experiment_id}",
            f"{'='*50}",
            f"Experiment: {self.metadata.name or self.experiment_id}",
            f"Anonymizer: {self.metadata.anonymizer or 'N/A'}",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Elapsed: {self.elapsed_seconds:.2f}s",
            f"",
        ]

        if self.success:
            lines.append(self.aggregated.summary())
        else:
            lines.append(f"Error: {self.error_message}")

        if self.output_files:
            lines.extend([
                "",
                "Output Files:",
            ])
            for name, path in self.output_files.items():
                lines.append(f"  {name}: {path}")

        return '\n'.join(lines)


class EvaluationPipeline:
    """
    Main evaluation pipeline orchestrator.

    Coordinates sample providers, evaluation modes, and output generation
    for complete evaluation workflows.

    Example:
        >>> # Simple usage
        >>> pipeline = EvaluationPipeline()
        >>> result = pipeline.run(provider)
        >>>
        >>> # Custom configuration
        >>> config = PipelineConfig(
        ...     mode='spherical',
        ...     metrics='full',
        ...     output_dir='./results'
        ... )
        >>> pipeline = EvaluationPipeline(config)
        >>> result = pipeline.run(provider)
        >>> print(result.summary())
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        self.config = config or PipelineConfig()
        self._mode: Optional[EvaluationMode] = None
        self._suite: Optional[MetricSuite] = None
        self._callbacks: List[Callable] = []

    def _create_metric_suite(self) -> MetricSuite:
        """Create metric suite from configuration."""
        metrics_config = self.config.metrics

        if metrics_config == "standard":
            return MetricSuite.standard()
        elif metrics_config == "full":
            return MetricSuite.full()
        elif isinstance(metrics_config, list):
            # Create suite from metric names
            from xdeid3d.metrics.registry import MetricRegistry

            metrics = []
            for name in metrics_config:
                metric = MetricRegistry.create(name)
                if metric:
                    metrics.append(metric)
            return MetricSuite(metrics, name="custom")
        else:
            return MetricSuite.standard()

    def _create_mode(self, suite: MetricSuite) -> EvaluationMode:
        """Create evaluation mode from configuration."""
        mode_name = self.config.mode.lower()

        if mode_name == "single":
            return SingleSampleMode(
                metric_suite=suite,
                fail_on_error=self.config.fail_on_error,
                verbose=self.config.verbose,
            )
        elif mode_name == "aggregate":
            return AggregateMode(
                metric_suite=suite,
                compute_confidence=self.config.compute_confidence,
                detect_outliers=self.config.detect_outliers,
                verbose=self.config.verbose,
            )
        elif mode_name == "spherical":
            return SphericalMode(
                metric_suite=suite,
                interpolation_bandwidth=self.config.interpolation_bandwidth,
                grid_resolution=self.config.grid_resolution,
                primary_metric=self.config.primary_metric,
                verbose=self.config.verbose,
            )
        else:
            # Default to aggregate
            return AggregateMode(
                metric_suite=suite,
                verbose=self.config.verbose,
            )

    def run(
        self,
        provider: SampleProvider,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        Run the evaluation pipeline.

        Args:
            provider: Sample provider to evaluate
            **kwargs: Additional arguments passed to evaluation

        Returns:
            PipelineResult with all outputs
        """
        start_time = time.perf_counter()
        experiment_id = str(uuid.uuid4())[:8]

        # Create output directory
        output_dir = None
        if self.config.output_dir:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=self.config.experiment_name or f"evaluation_{experiment_id}",
            description=self.config.experiment_description,
            config=self.config.to_dict(),
            anonymizer=self.config.anonymizer_name,
            num_samples=len(provider),
        )

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluation Pipeline: {metadata.name}")
            print(f"{'='*60}")
            print(f"Mode: {self.config.mode}")
            print(f"Metrics: {self.config.metrics}")
            print(f"Samples: {len(provider)}")
            print(f"{'='*60}\n")

        try:
            # Create metric suite and evaluation mode
            self._suite = self._create_metric_suite()
            self._mode = self._create_mode(self._suite)

            # Store metric names in metadata
            metadata.metrics = self._suite.metric_names

            # Run evaluation
            aggregated = self._mode.evaluate_provider(provider, **kwargs)

            # Save outputs
            output_files = {}
            if output_dir:
                output_files = self._save_outputs(
                    output_dir, metadata, aggregated, self._mode
                )

            # Mark completed
            metadata.mark_completed()

            elapsed = time.perf_counter() - start_time

            result = PipelineResult(
                experiment_id=experiment_id,
                metadata=metadata,
                aggregated=aggregated,
                output_dir=output_dir,
                output_files=output_files,
                elapsed_seconds=elapsed,
                success=True,
            )

            if self.config.verbose:
                print(f"\n{result.summary()}")

            return result

        except Exception as e:
            elapsed = time.perf_counter() - start_time

            return PipelineResult(
                experiment_id=experiment_id,
                metadata=metadata,
                aggregated=AggregatedResults(),
                output_dir=output_dir,
                elapsed_seconds=elapsed,
                success=False,
                error_message=str(e),
            )

    def _save_outputs(
        self,
        output_dir: Path,
        metadata: ExperimentMetadata,
        aggregated: AggregatedResults,
        mode: EvaluationMode,
    ) -> Dict[str, Path]:
        """Save evaluation outputs to files."""
        output_files = {}

        # Save metadata
        metadata_path = output_dir / "experiment_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        output_files['metadata'] = metadata_path

        # Save aggregated results
        if self.config.save_aggregated:
            aggregated_path = output_dir / "aggregated_results.json"
            aggregated.to_json(aggregated_path)
            output_files['aggregated'] = aggregated_path

        # Save per-sample results
        if self.config.save_per_sample and aggregated.per_sample_results:
            samples_path = output_dir / "per_sample_results.json"
            with open(samples_path, 'w') as f:
                json.dump(
                    [r.to_dict() for r in aggregated.per_sample_results],
                    f,
                    indent=2
                )
            output_files['per_sample'] = samples_path

        # Save spherical data if using spherical mode
        if isinstance(mode, SphericalMode):
            heatmap_path = output_dir / "spherical_heatmap.npz"
            yaw_grid, pitch_grid, score_grid = mode.get_spherical_heatmap()
            import numpy as np
            np.savez(
                heatmap_path,
                yaw=yaw_grid,
                pitch=pitch_grid,
                scores=score_grid
            )
            output_files['spherical_heatmap'] = heatmap_path

        # Generate report
        if self.config.generate_report:
            report_path = output_dir / "report.json"
            self._generate_report(report_path, metadata, aggregated)
            output_files['report'] = report_path

        return output_files

    def _generate_report(
        self,
        path: Path,
        metadata: ExperimentMetadata,
        aggregated: AggregatedResults,
    ) -> None:
        """Generate evaluation report."""
        report = {
            'experiment': metadata.to_dict(),
            'summary': {
                'total_samples': aggregated.total_samples,
                'successful_samples': aggregated.successful_samples,
                'failed_samples': aggregated.failed_samples,
                'total_time_ms': aggregated.total_time_ms,
            },
            'metrics': {},
        }

        for name, stats in aggregated.metric_stats.items():
            report['metrics'][name] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'median': stats.get('median'),
            }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

    def add_callback(
        self,
        callback: Callable[[EvaluationResult], None]
    ) -> None:
        """Add a callback to be called after each sample evaluation."""
        self._callbacks.append(callback)

    @classmethod
    def quick_evaluate(
        cls,
        provider: SampleProvider,
        mode: str = "aggregate",
        metrics: str = "standard",
        verbose: bool = True,
    ) -> AggregatedResults:
        """
        Quick evaluation without full pipeline setup.

        Args:
            provider: Sample provider
            mode: Evaluation mode
            metrics: Metric suite
            verbose: Print progress

        Returns:
            AggregatedResults
        """
        config = PipelineConfig(
            mode=mode,
            metrics=metrics,
            verbose=verbose,
            generate_report=False,
        )
        pipeline = cls(config)
        result = pipeline.run(provider)
        return result.aggregated

    @classmethod
    def from_config_file(cls, path: Union[str, Path]) -> "EvaluationPipeline":
        """Create pipeline from JSON/YAML config file."""
        path = Path(path)

        if path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(path) as f:
                    config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        else:
            with open(path) as f:
                config_dict = json.load(f)

        config = PipelineConfig(**config_dict)
        return cls(config)
