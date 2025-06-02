from __future__ import annotations

"""
Common types and utilities for the graph module.

This module provides common types and utilities to avoid circular imports
between memory_store and dependency_graph_builder.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Protocol for memory store to avoid circular imports."""

    def get_document(self, document_id: str) -> Any:
        """Get a document by ID."""
        ...

    def add_document(
        self, content: str, metadata: Any = None, document_id: str = None, embedding: Any = None
    ) -> Any:
        """Add a document to the memory store."""
        ...

    def update_document(
        self, document_id: str, content: str = None, metadata: Any = None, embedding: Any = None
    ) -> Any:
        """Update a document in the memory store."""
        ...

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the memory store."""
        ...

    def search(
        self, query_embedding: Any, limit: int = 10, filter_dict: Any = None, **kwargs
    ) -> list[tuple[Any, float]]:
        """Search for documents similar to the query embedding."""
        ...
