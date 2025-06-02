from __future__ import annotations

"""
Vector Store module for Saplings.

This module re-exports the public API from saplings.api.vector_store.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides vector store functionality for Saplings.
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.vector_store.

__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
    "FaissVectorStore",  # May not be available at runtime
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        from saplings.api.vector_store import (
            InMemoryVectorStore,
            VectorStore,
            get_vector_store,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "VectorStore": VectorStore,
            "InMemoryVectorStore": InMemoryVectorStore,
            "get_vector_store": get_vector_store,
        }

        # Try to import FaissVectorStore if requested
        if name == "FaissVectorStore":
            try:
                from saplings.api.vector_store import FaissVectorStore

                globals_dict["FaissVectorStore"] = FaissVectorStore
            except (ImportError, AttributeError):
                raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

        # Return the requested attribute
        return globals_dict.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
