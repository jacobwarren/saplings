from __future__ import annotations

"""
Vector store base implementation for Saplings.

This module provides the base vector store interface for Saplings.
"""


# Import the abstract base class to avoid circular imports
from saplings.vector_store._internal.base.vector_store_abc import VectorStoreABC

# Type alias for backward compatibility
VectorStore = VectorStoreABC


def get_vector_store(name: str = "in_memory") -> VectorStore:
    """
    Get a vector store by name.

    Args:
    ----
        name: Name of the vector store to get

    Returns:
    -------
        VectorStore: The vector store

    Raises:
    ------
        ValueError: If the vector store is not found

    """
    # Use factory functions to avoid circular imports
    if name == "in_memory":
        # Import at runtime to avoid circular imports
        from saplings.vector_store._internal.providers import create_in_memory_store

        return create_in_memory_store()
    elif name == "faiss":
        try:
            # Import at runtime to avoid circular imports
            from saplings.vector_store._internal.providers import create_faiss_store

            return create_faiss_store()
        except ImportError:
            msg = "FAISS is not installed. Install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'"
            raise ImportError(msg)
    else:
        msg = f"Vector store {name} not found"
        raise ValueError(msg)
