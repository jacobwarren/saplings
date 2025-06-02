from __future__ import annotations

"""
Vector Store API module for Saplings.

This module provides the public API for vector stores and related components.
"""

from typing import TypeVar, cast

from saplings.api.stability import beta, stable
from saplings.vector_store._internal import (
    InMemoryVectorStore as _InMemoryVectorStore,
)
from saplings.vector_store._internal import (
    VectorStore as _VectorStore,
)

# Conditionally import FAISS vector store if available
try:
    from saplings.vector_store._internal import FaissVectorStore as _FaissVectorStore

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    _FaissVectorStore = None

# Type variable for vector store classes
T = TypeVar("T", bound=_VectorStore)


@stable
class VectorStore(_VectorStore):
    """
    Base class for vector stores.

    This class defines the interface for vector stores, which are responsible for
    storing and retrieving document embeddings.
    """


@stable
class InMemoryVectorStore(_InMemoryVectorStore):
    """
    In-memory vector store.

    This vector store keeps all embeddings in memory, making it fast but not
    suitable for large datasets or persistence across restarts.
    """


# Conditionally define FaissVectorStore if FAISS is available
if HAS_FAISS and _FaissVectorStore is not None:

    @beta
    class FaissVectorStore(_FaissVectorStore):
        """
        FAISS-based vector store.

        This vector store uses Facebook AI Similarity Search (FAISS) for efficient
        similarity search in high-dimensional spaces. It supports both CPU and GPU
        acceleration for fast retrieval of similar vectors.
        """

# Note: FaissVectorStore is only available when FAISS is installed


@beta
def get_vector_store(name: str = "in_memory") -> VectorStore:
    """
    Get a vector store instance by name.

    Args:
    ----
        name: Name of the vector store to get

    Returns:
    -------
        The vector store instance

    Raises:
    ------
        ValueError: If the vector store type is not supported

    """
    if name == "in_memory":
        return cast(VectorStore, InMemoryVectorStore())
    elif name == "faiss":
        if not HAS_FAISS:
            msg = "FAISS is not installed. Install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'"
            raise ImportError(msg)
        return cast(VectorStore, FaissVectorStore())
    else:
        msg = f"Vector store {name} not found"
        raise ValueError(msg)


# Define __all__ to control what's imported with "from saplings.api.vector_store import *"
__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
]

# Add FaissVectorStore to __all__ if available
if HAS_FAISS:
    __all__.append("FaissVectorStore")
