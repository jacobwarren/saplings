from __future__ import annotations

"""
Vector store module for memory components.

This module provides vector store functionality for the Saplings framework.
"""

from saplings.memory._internal.vector_store.get_vector_store import get_vector_store
from saplings.memory._internal.vector_store.in_memory_vector_store import InMemoryVectorStore
from saplings.memory._internal.vector_store.vector_store import VectorStore

__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
]
