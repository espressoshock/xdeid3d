"""
Data structures for evaluation results and samples.

This module provides core data structures for organizing evaluation
inputs, outputs, and aggregated results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import json
import numpy as np

__all__ = [
    "EvaluationSample",
    "EvaluationResult",
    "AggregatedResults",
    "ExperimentMetadata",
    "EvaluationStatus",
    "SampleType",
]


class SampleType(str, Enum):
    """Type of evaluation sample."""
    IMAGE = "image"
    VIDEO = "video"
    FRAME_SEQUENCE = "frame_sequence"


class EvaluationStatus(str, Enum):
    """Status of an evaluation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class EvaluationSample:
    """
    A single sample for evaluation.

    Represents an original-anonymized pair with associated metadata
    like pose, frame index, and camera parameters.

    Attributes:
        sample_id: Unique identifier for the sample
        original: Original image/frame (np.ndarray or path)
        anonymized: Anonymized image/frame (np.ndarray or path)
        sample_type: Type of sample (image, video, frame_sequence)
        metadata: Additional sample metadata

    Example:
        >>> sample = EvaluationSample(
        ...     sample_id="frame_001",
        ...     original=original_image,
        ...     anonymized=anonymized_image,
        ...     metadata={'yaw': 1.57, 'pitch': 1.57}
        ... )
    """
    sample_id: str
    original: Union[np.ndarray, str, Path]
    anonymized: Union[np.ndarray, str, Path]
    sample_type: SampleType = SampleType.IMAGE
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields for video/sequence samples
    frame_index: Optional[int] = None
    timestamp_ms: Optional[float] = None

    # Pose information
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    # Depth map (for 3D analysis)
    depth: Optional[Union[np.ndarray, str, Path]] = None

    def load_images(self) -> "EvaluationSample":
        """Load images if they are paths."""
        import cv2

        def load_if_path(img):
            if isinstance(img, (str, Path)):
                loaded = cv2.imread(str(img))
                if loaded is not None:
                    return cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
                return None
            return img

        self.original = load_if_path(self.original)
        self.anonymized = load_if_path(self.anonymized)
        if self.depth is not None:
            self.depth = load_if_path(self.depth)

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (images as paths only)."""
        def path_or_shape(x):
            if isinstance(x, (str, Path)):
                return str(x)
            elif isinstance(x, np.ndarray):
                return f"<array {x.shape}>"
            return None

        return {
            'sample_id': self.sample_id,
            'original': path_or_shape(self.original),
            'anonymized': path_or_shape(self.anonymized),
            'sample_type': self.sample_type.value,
            'metadata': self.metadata,
            'frame_index': self.frame_index,
            'timestamp_ms': self.timestamp_ms,
            'yaw': self.yaw,
            'pitch': self.pitch,
            'roll': self.roll,
            'depth': path_or_shape(self.depth),
        }


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single sample or batch.

    Contains metric values, timing information, and any warnings
    or errors encountered during evaluation.

    Attributes:
        sample_id: ID of the evaluated sample
        metrics: Dictionary of metric name -> value
        status: Evaluation status
        processing_time_ms: Time taken to process
        warnings: List of warning messages
        errors: List of error messages
        raw_data: Raw metric data (histograms, maps, etc.)

    Example:
        >>> result = EvaluationResult(
        ...     sample_id="frame_001",
        ...     metrics={'arcface_distance': 0.85, 'ssim': 0.92},
        ...     status=EvaluationStatus.COMPLETED
        ... )
    """
    sample_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    status: EvaluationStatus = EvaluationStatus.PENDING
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    # Sample metadata (copied from sample)
    frame_index: Optional[int] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None

    def is_success(self) -> bool:
        """Check if evaluation completed successfully."""
        return self.status == EvaluationStatus.COMPLETED and len(self.errors) == 0

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a metric value by name."""
        return self.metrics.get(name, default)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error message and mark as failed."""
        self.errors.append(message)
        self.status = EvaluationStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sample_id': self.sample_id,
            'metrics': self.metrics,
            'status': self.status.value,
            'processing_time_ms': self.processing_time_ms,
            'warnings': self.warnings,
            'errors': self.errors,
            'frame_index': self.frame_index,
            'yaw': self.yaw,
            'pitch': self.pitch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(
            sample_id=data['sample_id'],
            metrics=data.get('metrics', {}),
            status=EvaluationStatus(data.get('status', 'completed')),
            processing_time_ms=data.get('processing_time_ms', 0.0),
            warnings=data.get('warnings', []),
            errors=data.get('errors', []),
            frame_index=data.get('frame_index'),
            yaw=data.get('yaw'),
            pitch=data.get('pitch'),
        )


@dataclass
class AggregatedResults:
    """
    Aggregated evaluation results across multiple samples.

    Provides statistics, distributions, and summaries for each metric
    computed over a set of samples.

    Attributes:
        metric_stats: Per-metric statistics (mean, std, min, max, etc.)
        per_sample_results: List of individual sample results
        total_samples: Total number of samples evaluated
        successful_samples: Number of successfully evaluated samples
        total_time_ms: Total processing time

    Example:
        >>> aggregated = AggregatedResults.from_results(results)
        >>> print(aggregated.metric_stats['arcface_distance'])
        {'mean': 0.82, 'std': 0.05, 'min': 0.71, 'max': 0.94}
    """
    metric_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_sample_results: List[EvaluationResult] = field(default_factory=list)
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    skipped_samples: int = 0
    total_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_results(cls, results: Sequence[EvaluationResult]) -> "AggregatedResults":
        """
        Create aggregated results from a list of individual results.

        Args:
            results: Sequence of EvaluationResult objects

        Returns:
            AggregatedResults with computed statistics
        """
        if not results:
            return cls()

        # Count statuses
        successful = sum(1 for r in results if r.status == EvaluationStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == EvaluationStatus.FAILED)
        skipped = sum(1 for r in results if r.status == EvaluationStatus.SKIPPED)

        # Compute per-metric statistics
        metric_values: Dict[str, List[float]] = {}
        for result in results:
            if result.status == EvaluationStatus.COMPLETED:
                for name, value in result.metrics.items():
                    if name not in metric_values:
                        metric_values[name] = []
                    if not np.isnan(value) and not np.isinf(value):
                        metric_values[name].append(value)

        metric_stats = {}
        for name, values in metric_values.items():
            if values:
                arr = np.array(values)
                metric_stats[name] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr)),
                    'count': len(values),
                    'percentile_25': float(np.percentile(arr, 25)),
                    'percentile_75': float(np.percentile(arr, 75)),
                }

        total_time = sum(r.processing_time_ms for r in results)

        return cls(
            metric_stats=metric_stats,
            per_sample_results=list(results),
            total_samples=len(results),
            successful_samples=successful,
            failed_samples=failed,
            skipped_samples=skipped,
            total_time_ms=total_time,
        )

    def get_metric_mean(self, name: str, default: float = 0.0) -> float:
        """Get mean value for a metric."""
        stats = self.metric_stats.get(name, {})
        return stats.get('mean', default)

    def get_metric_std(self, name: str, default: float = 0.0) -> float:
        """Get standard deviation for a metric."""
        stats = self.metric_stats.get(name, {})
        return stats.get('std', default)

    def get_metric_values(self, name: str) -> List[float]:
        """Get all values for a specific metric."""
        values = []
        for result in self.per_sample_results:
            if result.status == EvaluationStatus.COMPLETED:
                if name in result.metrics:
                    values.append(result.metrics[name])
        return values

    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            f"Evaluation Summary",
            f"{'='*40}",
            f"Total samples: {self.total_samples}",
            f"Successful: {self.successful_samples}",
            f"Failed: {self.failed_samples}",
            f"Skipped: {self.skipped_samples}",
            f"Total time: {self.total_time_ms:.1f}ms",
            f"",
            f"Metric Statistics:",
            f"{'-'*40}",
        ]

        for name, stats in sorted(self.metric_stats.items()):
            lines.append(
                f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"[{stats['min']:.4f}, {stats['max']:.4f}]"
            )

        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_stats': self.metric_stats,
            'per_sample_results': [r.to_dict() for r in self.per_sample_results],
            'total_samples': self.total_samples,
            'successful_samples': self.successful_samples,
            'failed_samples': self.failed_samples,
            'skipped_samples': self.skipped_samples,
            'total_time_ms': self.total_time_ms,
            'timestamp': self.timestamp,
        }

    def to_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "AggregatedResults":
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            metric_stats=data.get('metric_stats', {}),
            per_sample_results=[
                EvaluationResult.from_dict(r)
                for r in data.get('per_sample_results', [])
            ],
            total_samples=data.get('total_samples', 0),
            successful_samples=data.get('successful_samples', 0),
            failed_samples=data.get('failed_samples', 0),
            skipped_samples=data.get('skipped_samples', 0),
            total_time_ms=data.get('total_time_ms', 0.0),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
        )


@dataclass
class ExperimentMetadata:
    """
    Metadata for an evaluation experiment.

    Contains configuration, environment info, and experiment parameters.

    Attributes:
        experiment_id: Unique experiment identifier
        name: Human-readable experiment name
        description: Experiment description
        config: Configuration dictionary
        anonymizer: Anonymizer name/type used
        metrics: List of metrics evaluated
        created_at: Creation timestamp
        completed_at: Completion timestamp
    """
    experiment_id: str
    name: str = ""
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    anonymizer: str = ""
    metrics: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Dataset info
    dataset_name: str = ""
    dataset_path: str = ""
    num_samples: int = 0

    # Environment
    device: str = "cpu"
    seed: Optional[int] = None

    def mark_completed(self) -> None:
        """Mark experiment as completed."""
        self.completed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'anonymizer': self.anonymizer,
            'metrics': self.metrics,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'dataset_name': self.dataset_name,
            'dataset_path': self.dataset_path,
            'num_samples': self.num_samples,
            'device': self.device,
            'seed': self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetadata":
        """Create from dictionary."""
        return cls(
            experiment_id=data['experiment_id'],
            name=data.get('name', ''),
            description=data.get('description', ''),
            config=data.get('config', {}),
            anonymizer=data.get('anonymizer', ''),
            metrics=data.get('metrics', []),
            created_at=data.get('created_at', datetime.now().isoformat()),
            completed_at=data.get('completed_at'),
            dataset_name=data.get('dataset_name', ''),
            dataset_path=data.get('dataset_path', ''),
            num_samples=data.get('num_samples', 0),
            device=data.get('device', 'cpu'),
            seed=data.get('seed'),
        )
