from __future__ import annotations

"""
Memory Indexer API module for Saplings.

This module provides the public API for memory indexers.
"""

from saplings.api.stability import beta, stable
from saplings.memory._internal.indexer import Indexer as _Indexer
from saplings.memory._internal.indexer import SimpleIndexer as _SimpleIndexer


@stable
class Indexer(_Indexer):
    """
    Base class for indexers.

    This class defines the interface for indexers, which are responsible for
    indexing documents in the memory store.
    """


@stable
class SimpleIndexer(_SimpleIndexer):
    """
    Simple indexer for memory store documents.

    This indexer provides basic indexing functionality for memory store documents,
    including entity extraction and relationship identification.

    It is designed to be simple and efficient, with minimal dependencies.
    """


# Import registry functionality
from saplings.memory._internal.registry import (
    IndexerRegistry as _IndexerRegistry,
)
from saplings.memory._internal.registry import (
    get_indexer as _get_indexer,
)


@stable
class IndexerRegistry(_IndexerRegistry):
    """
    Registry for indexers.

    This class provides a registry for indexers, allowing them to be registered
    and retrieved by name.
    """


@beta
def get_indexer(name: str) -> Indexer:
    """
    Get an indexer by name.

    Args:
    ----
        name: Name of the indexer to get

    Returns:
    -------
        The indexer instance

    Raises:
    ------
        ValueError: If the indexer is not found

    """
    # Get the indexer from the internal registry
    indexer = _get_indexer(name)
    return indexer


__all__ = [
    "Indexer",
    "IndexerRegistry",
    "SimpleIndexer",
    "get_indexer",
]
