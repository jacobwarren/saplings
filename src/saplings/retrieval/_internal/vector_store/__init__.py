from __future__ import annotations

"""
Vector store module for retrieval components.

This module provides vector storage implementations for the Saplings framework.
"""

from saplings.retrieval._internal.vector_store.faiss_vector_store import FaissVectorStore

__all__ = [
    "FaissVectorStore",
]
