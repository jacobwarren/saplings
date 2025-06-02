from __future__ import annotations

"""
Memory Store API module for Saplings.

This module provides the public API for memory stores.
"""


from saplings.api.stability import beta, stable
from saplings.memory._internal.config import MemoryConfig as _MemoryConfig
from saplings.memory._internal.memory_store import MemoryStore as _MemoryStore


@stable
class MemoryConfig(_MemoryConfig):
    """
    Configuration for memory.

    This class provides configuration options for memory, including
    vector store type, chunk size, and chunk overlap.
    """


@stable
class MemoryStore(_MemoryStore):
    """
    Memory store for documents.

    This class provides a store for documents, with support for
    vector search, graph search, and hybrid search.
    """


@beta
class MemoryStoreBuilder:
    """
    Builder for memory stores.

    This class provides a fluent interface for building memory stores.
    """

    def __init__(self) -> None:
        """
        Initialize the builder.
        """
        self._config = MemoryConfig()

    def with_vector_store_type(self, vector_store_type: str) -> "MemoryStoreBuilder":
        """
        Set the vector store type.

        Args:
        ----
            vector_store_type: Type of vector store to use

        Returns:
        -------
            Self for chaining

        """
        self._config.vector_store_type = vector_store_type
        return self

    def with_chunk_size(self, chunk_size: int) -> "MemoryStoreBuilder":
        """
        Set the chunk size.

        Args:
        ----
            chunk_size: Size of chunks in tokens

        Returns:
        -------
            Self for chaining

        """
        self._config.chunk_size = chunk_size
        return self

    def with_chunk_overlap(self, chunk_overlap: int) -> "MemoryStoreBuilder":
        """
        Set the chunk overlap.

        Args:
        ----
            chunk_overlap: Overlap between chunks in tokens

        Returns:
        -------
            Self for chaining

        """
        self._config.chunk_overlap = chunk_overlap
        return self

    def with_enable_graph(self, enable_graph: bool) -> "MemoryStoreBuilder":
        """
        Set whether to enable graph storage.

        Args:
        ----
            enable_graph: Whether to enable graph storage

        Returns:
        -------
            Self for chaining

        """
        self._config.enable_graph = enable_graph
        return self

    def build(self) -> MemoryStore:
        """
        Build the memory store.

        Returns
        -------
            The built memory store

        """
        return MemoryStore(self._config)


__all__ = [
    "MemoryConfig",
    "MemoryStore",
    "MemoryStoreBuilder",
]
