from __future__ import annotations

"""
Base module for vector store components.

This module provides the base interface for vector stores in the Saplings framework.
"""

from saplings.vector_store._internal.base.vector_store import VectorStore, get_vector_store

__all__ = [
    "VectorStore",
    "get_vector_store",
]
