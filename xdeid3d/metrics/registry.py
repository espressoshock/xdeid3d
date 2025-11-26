"""
Metric registry for plugin discovery and management.

This module provides a registry pattern for discovering and instantiating
metrics by name, supporting both built-in metrics and third-party plugins.
"""

from importlib.metadata import entry_points
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
import warnings

from xdeid3d.metrics.base import BaseMetric, MetricCategory

__all__ = [
    "MetricRegistry",
    "register_metric",
]


T = TypeVar("T", bound=BaseMetric)


class MetricRegistry:
    """
    Registry for discovering and instantiating metrics.

    The registry supports:
    - Built-in metrics registered via decorator
    - Third-party metrics registered via entry_points
    - Runtime registration via register() method

    Example:
        >>> @MetricRegistry.register("psnr")
        ... class PSNRMetric(BaseMetric):
        ...     pass
        >>>
        >>> metric = MetricRegistry.get("psnr")
        >>> result = metric.compute(original, anonymized)
    """

    # Class-level storage
    _registry: Dict[str, Type[BaseMetric]] = {}
    _entry_points_loaded: bool = False
    _entry_point_group: str = "xdeid3d.metrics"

    @classmethod
    def register(
        cls, name: Optional[str] = None
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a metric class.

        Args:
            name: Name to register under (defaults to class name)

        Returns:
            Decorator function
        """

        def decorator(metric_cls: Type[T]) -> Type[T]:
            register_name = name or metric_cls.__name__
            cls._registry[register_name.lower()] = metric_cls
            return metric_cls

        return decorator

    @classmethod
    def register_class(
        cls,
        metric_cls: Type[BaseMetric],
        name: Optional[str] = None,
    ) -> None:
        """Register a metric class directly."""
        register_name = (name or metric_cls.__name__).lower()
        cls._registry[register_name] = metric_cls

    @classmethod
    def _load_entry_points(cls) -> None:
        """Load metrics from entry_points (third-party plugins)."""
        if cls._entry_points_loaded:
            return

        try:
            eps = entry_points(group=cls._entry_point_group)
        except TypeError:
            eps = entry_points().get(cls._entry_point_group, [])

        for ep in eps:
            try:
                metric_cls = ep.load()
                cls._registry[ep.name.lower()] = metric_cls
            except Exception as e:
                warnings.warn(
                    f"Failed to load metric entry point '{ep.name}': {e}",
                    RuntimeWarning,
                )

        cls._entry_points_loaded = True

    @classmethod
    def get(
        cls,
        name: str,
        **kwargs: Any,
    ) -> BaseMetric:
        """
        Get a metric instance by name.

        Args:
            name: Metric name (case-insensitive)
            **kwargs: Arguments passed to metric constructor

        Returns:
            Instantiated metric

        Raises:
            KeyError: If metric not found
        """
        cls._load_entry_points()

        name_lower = name.lower()

        if name_lower not in cls._registry:
            available = ", ".join(cls.list_available())
            raise KeyError(
                f"Metric '{name}' not found. Available: {available or 'none'}"
            )

        metric_cls = cls._registry[name_lower]
        return metric_cls(**kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type[BaseMetric]:
        """Get a metric class by name (without instantiating)."""
        cls._load_entry_points()

        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise KeyError(f"Metric '{name}' not found")

        return cls._registry[name_lower]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available metric names."""
        cls._load_entry_points()
        return sorted(cls._registry.keys())

    @classmethod
    def list_by_category(cls, category: MetricCategory) -> List[str]:
        """
        List metrics by category.

        Args:
            category: Metric category to filter by

        Returns:
            List of metric names in the category
        """
        cls._load_entry_points()

        result = []
        for name, metric_cls in cls._registry.items():
            try:
                # Try to instantiate to check category
                metric = metric_cls()
                if metric.category == category:
                    result.append(name)
            except Exception:
                # Skip metrics that fail to instantiate
                pass

        return sorted(result)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a metric is registered."""
        cls._load_entry_points()
        return name.lower() in cls._registry

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a metric."""
        name_lower = name.lower()
        if name_lower in cls._registry:
            del cls._registry[name_lower]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics."""
        cls._registry.clear()
        cls._entry_points_loaded = False

    @classmethod
    def get_multiple(
        cls,
        names: List[str],
        **kwargs: Any,
    ) -> List[BaseMetric]:
        """
        Get multiple metric instances.

        Args:
            names: List of metric names
            **kwargs: Arguments passed to all metric constructors

        Returns:
            List of instantiated metrics
        """
        return [cls.get(name, **kwargs) for name in names]


# Convenience decorator alias
register_metric = MetricRegistry.register


def create_metric(name: str, **kwargs: Any) -> BaseMetric:
    """Convenience function to create a metric by name."""
    return MetricRegistry.get(name, **kwargs)
