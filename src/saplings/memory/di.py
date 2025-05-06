from __future__ import annotations

"""
Dependency injection module for Saplings memory components.

This module provides the factory functions and bindings for memory-related
components in the dependency injection container.
"""


from saplings.memory.config import MemoryConfig, VectorStoreType
from saplings.memory.vector_store import InMemoryVectorStore

# Try to import FAISS components
try:
    from saplings.retrieval.faiss_vector_store import FaissVectorStore

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def get_vector_store(config=None):
    """
    Factory function to create the appropriate vector store based on configuration.

    Args:
    ----
        config: MemoryConfig object or None

    Returns:
    -------
        A vector store instance

    """
    config = config or MemoryConfig.default()
    store_type = config.vector_store.store_type

    if store_type == VectorStoreType.FAISS:
        # Use FAISS if available and configured
        if HAS_FAISS:
            # Check for GPU configuration
            use_gpu = getattr(config.vector_store, "use_gpu", False)
            return FaissVectorStore(config, use_gpu=use_gpu)
        # Fall back to in-memory if FAISS is unavailable
        return InMemoryVectorStore(config)
    if store_type == VectorStoreType.IN_MEMORY:
        # Default to in-memory vector store
        return InMemoryVectorStore(config)
    # For other store types, delegate to the plugin-based resolver
    from saplings.memory.vector_store import get_vector_store as plugin_resolver

    return plugin_resolver(config)
