from __future__ import annotations

"""
Service locator for Saplings.

This module provides the service locator for Saplings.
"""

from typing import Any, Dict, Optional

# Import from common module to avoid circular dependencies


class ServiceLocator:
    """Service locator for Saplings."""
    
    _instance: Optional["ServiceLocator"] = None
    
    def __init__(self) -> None:
        """Initialize the service locator."""
        self._services: Dict[str, Any] = {}
    
    @classmethod
    def get_instance(cls) -> "ServiceLocator":
        """Get the singleton instance of the service locator."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register_service(self, name: str, service: Any) -> None:
        """Register a service."""
        self._services[name] = service
    
    def get_service(self, name: str) -> Any:
        """Get a service by name."""
        return self._services.get(name)


class IndexerRegistry:
    """Registry for indexers."""

    def __init__(self) -> None:
        """Initialize the indexer registry."""
        self._indexers: Dict[str, Any] = {}

    def register_indexer(self, name: str, indexer: Any) -> None:
        """Register an indexer."""
        self._indexers[name] = indexer

    def get_indexer(self, name: str) -> Any:
        """Get an indexer by name."""
        return self._indexers.get(name)


# Function to get the plugin registry from the service locator
def get_plugin_registry(registry: Optional[Any] = None) -> Any:
    """
    Get the plugin registry from the service locator.

    This is a forward declaration to avoid circular imports.
    The actual implementation is in core/registry.py.

    Args:
    ----
        registry: Optional explicit registry to use

    Returns:
    -------
        Any: The plugin registry

    """

    # Use a factory function to avoid circular imports
    def _get_registry():
        from saplings.registry._internal.core.registry import (
            get_plugin_registry as _get_plugin_registry,
        )

        return _get_plugin_registry(registry)

    # Call the factory function
    return _get_registry()
