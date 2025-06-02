from __future__ import annotations

"""
Registry module for Saplings memory.

This module defines the registry for memory components.
"""

import logging
from typing import Dict, Type, TypeVar

from saplings.memory._internal.indexer import Indexer

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for indexers
T = TypeVar("T", bound=Indexer)


class IndexerRegistry:
    """
    Registry for indexers.

    This class provides a registry for indexers, allowing them to be registered
    and retrieved by name.
    """

    _instance = None
    _indexers: Dict[str, Type[Indexer]] = {}

    def __new__(cls):
        """
        Create a new instance of the registry.

        Returns
        -------
            The singleton instance

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, name: str, indexer_cls: Type[T]) -> None:
        """
        Register an indexer.

        Args:
        ----
            name: Name to register the indexer under
            indexer_cls: Indexer class to register

        """
        if name in self._indexers:
            logger.warning(f"Overwriting existing indexer: {name}")
        self._indexers[name] = indexer_cls
        logger.debug(f"Registered indexer: {name}")

    def get(self, name: str) -> Type[Indexer]:
        """
        Get an indexer by name.

        Args:
        ----
            name: Name of the indexer to get

        Returns:
        -------
            The indexer class

        Raises:
        ------
            ValueError: If the indexer is not found

        """
        if name not in self._indexers:
            raise ValueError(f"Indexer not found: {name}")
        return self._indexers[name]

    def list(self) -> Dict[str, Type[Indexer]]:
        """
        List all registered indexers.

        Returns
        -------
            Dict of indexer names to indexer classes

        """
        return self._indexers.copy()


# Create a singleton instance
_registry = IndexerRegistry()

# Auto-register SimpleIndexer
try:
    from saplings.memory._internal.simple_indexer import SimpleIndexer

    _registry.register("simple", SimpleIndexer)
    logger.info("Auto-registered SimpleIndexer in singleton registry")
except Exception as e:
    logger.warning(f"Failed to auto-register SimpleIndexer in singleton registry: {e}")


def register_indexer(name: str, indexer_cls: Type[T]) -> Type[T]:
    """
    Register an indexer.

    This function can be used as a decorator:

    ```python
    @register_indexer("my_indexer")
    class MyIndexer(Indexer):
        pass
    ```

    Args:
    ----
        name: Name to register the indexer under
        indexer_cls: Indexer class to register

    Returns:
    -------
        The indexer class

    """
    _registry.register(name, indexer_cls)
    return indexer_cls


def get_indexer(name: str) -> Indexer:
    """
    Get an indexer by name.

    Args:
    ----
        name: Name of the indexer to get

    Returns:
    -------
        An instance of the indexer

    Raises:
    ------
        ValueError: If the indexer is not found

    """
    indexer_cls = _registry.get(name)
    return indexer_cls()


def list_indexers() -> Dict[str, Type[Indexer]]:
    """
    List all registered indexers.

    Returns
    -------
        Dict of indexer names to indexer classes

    """
    return _registry.list()
