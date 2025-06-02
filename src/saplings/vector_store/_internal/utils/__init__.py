from __future__ import annotations

"""
Utilities module for vector store components.

This module provides utility functions for vector operations in the Saplings framework.
"""

from saplings.vector_store._internal.utils.vector_utils import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    normalize_vectors,
)

__all__ = [
    "normalize_vectors",
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
]
