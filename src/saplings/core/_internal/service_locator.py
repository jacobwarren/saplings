from __future__ import annotations

"""
Service Locator module for Saplings.

This module provides a service locator pattern implementation that allows
accessing various registries without relying on the container. It serves as
a central registry for accessing services and provides a clean way to customize
behavior without monkey-patching.
"""

import logging
from typing import Any, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceLocator:
    """
    Service locator for Saplings.

    This class provides a central registry for accessing services and registries
    without relying on the container. It allows customizing behavior without
    monkey-patching and provides a clean way to override services for testing.

    Example:
    -------
    ```python
    # Get the service locator instance
    locator = ServiceLocator.get_instance()

    # Register a custom plugin registry
    locator.register("plugin_registry", my_plugin_registry)

    # Get the plugin registry
    plugin_registry = locator.get("plugin_registry")
    ```

    """

    _instance: Optional[ServiceLocator] = None
    _registries: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> ServiceLocator:
        """
        Get the singleton instance of the service locator.

        Returns
        -------
            ServiceLocator: The singleton instance

        """
        if cls._instance is None:
            cls._instance = ServiceLocator()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the service locator instance and clear all registries."""
        cls._instance = None
        cls._registries = {}
        logger.debug("Service locator reset")

    def register(self, registry_type: str, registry: Any) -> None:
        """
        Register a registry with the service locator.

        Args:
        ----
            registry_type: Type of registry (e.g., "plugin_registry", "indexer_registry")
            registry: Registry instance

        """
        self._registries[registry_type] = registry
        logger.debug(f"Registered {registry_type}")

    def get(self, registry_type: str, default_factory: Optional[callable] = None) -> Any:
        """
        Get a registry from the service locator.

        Args:
        ----
            registry_type: Type of registry to get
            default_factory: Optional factory function to create a default registry if not found

        Returns:
        -------
            The registry instance

        Raises:
        ------
            KeyError: If the registry is not found and no default factory is provided

        """
        if registry_type not in self._registries and default_factory is not None:
            # Create default registry using the factory
            self._registries[registry_type] = default_factory()
            logger.debug(f"Created default {registry_type}")

        if registry_type not in self._registries:
            msg = f"Registry not found: {registry_type}"
            raise KeyError(msg)

        return self._registries[registry_type]

    def has(self, registry_type: str) -> bool:
        """
        Check if a registry exists in the service locator.

        Args:
        ----
            registry_type: Type of registry to check

        Returns:
        -------
            bool: True if the registry exists, False otherwise

        """
        return registry_type in self._registries


class RegistryContext:
    """
    Context manager for temporary registry overrides.

    This class provides a context manager that temporarily overrides a registry
    in the service locator and restores the original registry when the context
    is exited.

    Example:
    -------
    ```python
    # Create a custom plugin registry
    my_plugin_registry = PluginRegistry()

    # Use the custom registry temporarily
    with RegistryContext("plugin_registry", my_plugin_registry):
        # Code in this block uses my_plugin_registry
        memory_store = MemoryStore(config=memory_config)

    # Code outside the block uses the original registry
    ```

    """

    def __init__(self, registry_type: str, registry: Any):
        """
        Initialize the registry context.

        Args:
        ----
            registry_type: Type of registry to override
            registry: Registry instance to use temporarily

        """
        self.registry_type = registry_type
        self.registry = registry
        self.original = None
        self.locator = ServiceLocator.get_instance()

    def __enter__(self) -> Any:
        """
        Enter the context and override the registry.

        Returns
        -------
            The temporary registry instance

        """
        # Save the original registry if it exists
        if self.locator.has(self.registry_type):
            self.original = self.locator.get(self.registry_type)

        # Register the temporary registry
        self.locator.register(self.registry_type, self.registry)
        return self.registry

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context and restore the original registry.

        Args:
        ----
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        """
        if self.original is not None:
            # Restore the original registry
            self.locator.register(self.registry_type, self.original)
        else:
            # Remove the temporary registry
            self._remove_registry(self.registry_type)

    def _remove_registry(self, registry_type: str) -> None:
        """
        Remove a registry from the service locator.

        Args:
        ----
            registry_type: Type of registry to remove

        """
        if registry_type in self.locator._registries:
            del self.locator._registries[registry_type]
