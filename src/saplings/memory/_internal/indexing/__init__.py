from __future__ import annotations

"""
Indexing module for memory components.

This module provides document indexing functionality for the Saplings framework.
"""

from saplings.memory._internal.indexing.entity import Entity
from saplings.memory._internal.indexing.indexer import Indexer
from saplings.memory._internal.indexing.indexer_registry import (
    IndexerRegistry,
    get_indexer,
    get_indexer_registry,
)
from saplings.memory._internal.indexing.indexing_result import IndexingResult
from saplings.memory._internal.indexing.simple_indexer import SimpleIndexer

__all__ = [
    "Entity",
    "Indexer",
    "IndexerRegistry",
    "IndexingResult",
    "SimpleIndexer",
    "get_indexer",
    "get_indexer_registry",
]
