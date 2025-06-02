from __future__ import annotations

"""
Providers module for vector store components.

This module provides vector store implementations for the Saplings framework.
"""

from typing import Any

# Import the actual classes
from saplings.vector_store._internal.providers.in_memory_vector_store import InMemoryVectorStore


# Factory functions to avoid circular imports
def create_in_memory_store() -> Any:
    """
    Create an in-memory vector store.

    Returns
    -------
        InMemoryVectorStore: An in-memory vector store

    """
    return InMemoryVectorStore()


def create_faiss_store() -> Any:
    """
    Create a FAISS vector store.

    Returns
    -------
        FaissVectorStore: A FAISS vector store

    Raises
    ------
        ImportError: If FAISS is not installed

    """

    # Use a factory function to avoid circular imports
    def factory():
        from saplings.vector_store._internal.providers.faiss_vector_store import FaissVectorStore

        return FaissVectorStore()

    return factory()


# Check if FAISS is available
try:
    import faiss  # type: ignore[import]

    from saplings.vector_store._internal.providers.faiss_vector_store import FaissVectorStore

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    FaissVectorStore = None

__all__ = [
    "InMemoryVectorStore",
    "create_in_memory_store",
    "create_faiss_store",
    "HAS_FAISS",
]

# Add FAISS vector store to __all__ if available
if HAS_FAISS:
    __all__.append("FaissVectorStore")
