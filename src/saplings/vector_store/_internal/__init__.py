from __future__ import annotations

"""
Internal implementation of the Vector Store module.

This module contains internal implementation details that are not part of the public API.
These components should not be used directly by application code.
"""

# Import from base
from saplings.vector_store._internal.base import (
    VectorStore,
    get_vector_store,
)

# Import from providers
from saplings.vector_store._internal.providers import (
    InMemoryVectorStore,
)

# Import from utils
from saplings.vector_store._internal.utils import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    normalize_vectors,
)

# Conditionally import FAISS vector store if available
try:
    from saplings.vector_store._internal.providers import FaissVectorStore

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

__all__ = [
    # Base
    "VectorStore",
    "get_vector_store",
    # Providers
    "InMemoryVectorStore",
    # Utils
    "normalize_vectors",
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
]

# Add FAISS vector store to __all__ if available
if HAS_FAISS:
    __all__.append("FaissVectorStore")
