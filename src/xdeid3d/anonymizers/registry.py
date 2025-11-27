"""
Anonymizer registry for plugin discovery and management.

This module provides a registry pattern for discovering and instantiating
anonymizers by name, supporting both built-in anonymizers and third-party
plugins via entry_points.
"""

from importlib.metadata import entry_points
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
import warnings

from xdeid3d.anonymizers.base import AnonymizerProtocol, BaseAnonymizer

__all__ = [
    "AnonymizerRegistry",
    "register_anonymizer",
]


T = TypeVar("T", bound=BaseAnonymizer)


class AnonymizerRegistry:
    """
    Registry for discovering and instantiating anonymizers.

    The registry supports:
    - Built-in anonymizers registered via decorator
    - Third-party anonymizers registered via entry_points
    - Runtime registration via register() method

    Example:
        >>> # Register via decorator
        >>> @AnonymizerRegistry.register("my_anonymizer")
        ... class MyAnonymizer(BaseAnonymizer):
        ...     pass
        >>>
        >>> # Get anonymizer by name
        >>> anonymizer = AnonymizerRegistry.get("my_anonymizer")
        >>>
        >>> # List all available anonymizers
        >>> print(AnonymizerRegistry.list_available())
    """

    # Class-level storage for registered anonymizers
    _registry: Dict[str, Type[BaseAnonymizer]] = {}
    _entry_points_loaded: bool = False
    _entry_point_group: str = "xdeid3d.anonymizers"

    @classmethod
    def register(
        cls, name: Optional[str] = None
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register an anonymizer class.

        Args:
            name: Name to register under (defaults to class name)

        Returns:
            Decorator function

        Example:
            >>> @AnonymizerRegistry.register("blur")
            ... class BlurAnonymizer(BaseAnonymizer):
            ...     pass
        """

        def decorator(anonymizer_cls: Type[T]) -> Type[T]:
            register_name = name or anonymizer_cls.__name__
            cls._registry[register_name.lower()] = anonymizer_cls
            return anonymizer_cls

        return decorator

    @classmethod
    def register_class(
        cls,
        anonymizer_cls: Type[BaseAnonymizer],
        name: Optional[str] = None,
    ) -> None:
        """
        Register an anonymizer class directly.

        Args:
            anonymizer_cls: Anonymizer class to register
            name: Name to register under (defaults to class name)
        """
        register_name = (name or anonymizer_cls.__name__).lower()
        cls._registry[register_name] = anonymizer_cls

    @classmethod
    def _load_entry_points(cls) -> None:
        """Load anonymizers from entry_points (third-party plugins)."""
        if cls._entry_points_loaded:
            return

        try:
            # Python 3.10+ style
            eps = entry_points(group=cls._entry_point_group)
        except TypeError:
            # Python 3.9 style
            eps = entry_points().get(cls._entry_point_group, [])

        for ep in eps:
            try:
                anonymizer_cls = ep.load()
                cls._registry[ep.name.lower()] = anonymizer_cls
            except Exception as e:
                warnings.warn(
                    f"Failed to load anonymizer entry point '{ep.name}': {e}",
                    RuntimeWarning,
                )

        cls._entry_points_loaded = True

    @classmethod
    def get(
        cls,
        name: str,
        **kwargs: Any,
    ) -> BaseAnonymizer:
        """
        Get an anonymizer instance by name.

        Args:
            name: Anonymizer name (case-insensitive)
            **kwargs: Arguments passed to anonymizer constructor

        Returns:
            Instantiated anonymizer

        Raises:
            KeyError: If anonymizer not found
        """
        cls._load_entry_points()

        name_lower = name.lower()

        if name_lower not in cls._registry:
            available = ", ".join(cls.list_available())
            raise KeyError(
                f"Anonymizer '{name}' not found. "
                f"Available: {available or 'none'}"
            )

        anonymizer_cls = cls._registry[name_lower]
        return anonymizer_cls(**kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type[BaseAnonymizer]:
        """
        Get an anonymizer class by name (without instantiating).

        Args:
            name: Anonymizer name (case-insensitive)

        Returns:
            Anonymizer class

        Raises:
            KeyError: If anonymizer not found
        """
        cls._load_entry_points()

        name_lower = name.lower()

        if name_lower not in cls._registry:
            raise KeyError(f"Anonymizer '{name}' not found")

        return cls._registry[name_lower]

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available anonymizer names.

        Returns:
            List of registered anonymizer names
        """
        cls._load_entry_points()
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an anonymizer is registered.

        Args:
            name: Anonymizer name

        Returns:
            True if registered
        """
        cls._load_entry_points()
        return name.lower() in cls._registry

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister an anonymizer.

        Args:
            name: Anonymizer name

        Returns:
            True if successfully unregistered, False if not found
        """
        name_lower = name.lower()
        if name_lower in cls._registry:
            del cls._registry[name_lower]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered anonymizers (mainly for testing)."""
        cls._registry.clear()
        cls._entry_points_loaded = False

    @classmethod
    def from_config(
        cls,
        config: Union[Dict[str, Any], "AnonymizerConfig"],
    ) -> BaseAnonymizer:
        """
        Create anonymizer from configuration.

        Args:
            config: Configuration dict or AnonymizerConfig object

        Returns:
            Instantiated anonymizer
        """
        from xdeid3d.config import AnonymizerConfig

        if isinstance(config, AnonymizerConfig):
            name = config.name or config.type
            kwargs = config.config
        else:
            name = config.get("name") or config.get("type", "blur")
            kwargs = config.get("config", {})

        return cls.get(name, **kwargs)


# Convenience decorator alias
register_anonymizer = AnonymizerRegistry.register


def create_anonymizer(
    name: str,
    **kwargs: Any,
) -> BaseAnonymizer:
    """
    Convenience function to create an anonymizer by name.

    Args:
        name: Anonymizer name
        **kwargs: Arguments passed to anonymizer constructor

    Returns:
        Instantiated anonymizer
    """
    return AnonymizerRegistry.get(name, **kwargs)
