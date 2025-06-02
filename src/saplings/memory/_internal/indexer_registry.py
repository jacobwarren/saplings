from __future__ import annotations

"""
Indexer registry module for Saplings memory.

This module defines the IndexerRegistry class for managing indexers.
"""

import logging
from typing import TYPE_CHECKING

# Import from other internal modules
from saplings.memory._internal.config import MemoryConfig

# Avoid circular import by not importing inject at module level
# from saplings.di import inject

if TYPE_CHECKING:
    from saplings.memory._internal.indexer import Indexer


# Configure logging
logger = logging.getLogger(__name__)


class IndexerRegistry:
    """
    Registry for indexers.

    This class manages the available indexers and provides methods to get
    indexers by name.
    """

    def __init__(self) -> None:
        """Initialize the indexer registry."""
        self._indexers = {}

    def register_indexer(self, name: str, indexer_class: type) -> None:
        """
        Register an indexer.

        Args:
        ----
            name: Name of the indexer
            indexer_class: Indexer class

        """
        # Import here to avoid circular imports
        from saplings.memory._internal.indexer import Indexer

        if not issubclass(indexer_class, Indexer):
            msg = f"Indexer class must be a subclass of Indexer, got {indexer_class}"
            raise TypeError(msg)

        self._indexers[name] = indexer_class
        logger.info(f"Registered indexer: {name}")

    def get_indexer(self, name: str, config: MemoryConfig | None = None) -> "Indexer":
        """
        Get an indexer by name.

        Args:
        ----
            name: Name of the indexer
            config: Memory configuration

        Returns:
        -------
            Indexer: Indexer instance

        Raises:
        ------
            ValueError: If the indexer is not found

        """
        if name not in self._indexers:
            msg = f"Indexer not found: {name}"
            raise ValueError(msg)

        indexer_class = self._indexers[name]
        return indexer_class(config)

    def list_indexers(self):
        """
        List available indexers.

        Returns
        -------
            List[str]: List of indexer names

        """
        return list(self._indexers.keys())


# Register the registry with the DI container
# This will be done in the public API module to avoid circular imports


# Global registry instance
_global_registry = None


# Function to get the registry from the container
def get_indexer_registry(registry: IndexerRegistry | None = None) -> IndexerRegistry:
    """
    Get the indexer registry.

    Args:
    ----
        registry: Optional registry to use

    Returns:
    -------
        IndexerRegistry: Indexer registry

    """
    global _global_registry

    # If a registry is provided, use it
    if registry is not None:
        return registry

    # Use global registry if available
    if _global_registry is not None:
        return _global_registry

    # Otherwise, create a new registry and register SimpleIndexer
    _global_registry = IndexerRegistry()

    # Automatically register SimpleIndexer
    try:
        from saplings.memory._internal.simple_indexer import SimpleIndexer

        _global_registry.register_indexer("simple", SimpleIndexer)
        logger.info("Automatically registered SimpleIndexer")
    except Exception as e:
        logger.warning(f"Failed to auto-register SimpleIndexer: {e}")

    return _global_registry


# Register the SimpleIndexer
def register_simple_indexer(registry: IndexerRegistry | None = None):
    """
    Register the SimpleIndexer.

    Args:
    ----
        registry: Optional indexer registry to use

    """
    try:
        # Import here to avoid circular imports
        from saplings.memory._internal.simple_indexer import SimpleIndexer

        indexer_registry = get_indexer_registry(registry)
        indexer_registry.register_indexer("simple", SimpleIndexer)
    except Exception as e:
        # Log the error but don't fail - this will be retried later
        logger.warning(f"Failed to register SimpleIndexer: {e}")


# Function to get an indexer by name
def get_indexer(
    name: str = "simple",
    config: MemoryConfig | None = None,
    registry: IndexerRegistry | None = None,
) -> "Indexer":
    """
    Get an indexer by name.

    Args:
    ----
        name: Name of the indexer
        config: Memory configuration
        registry: Optional indexer registry to use

    Returns:
    -------
        Indexer: Indexer instance

    """
    # Register the SimpleIndexer if needed
    if name == "simple":
        register_simple_indexer(registry)

    # Get the indexer from the registry
    return get_indexer_registry(registry).get_indexer(name, config)
