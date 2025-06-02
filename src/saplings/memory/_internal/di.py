from __future__ import annotations

"""
Dependency injection module for Saplings memory components.

This module provides the factory functions and bindings for memory-related
components in the dependency injection container.
"""

import importlib

from saplings.memory._internal.config import MemoryConfig, VectorStoreType
from saplings.memory._internal.vector_store import InMemoryVectorStore


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
        # Use lazy import to avoid circular dependency
        # Only import FaissVectorStore when needed
        try:
            # Import from the public API
            retrieval_module = importlib.import_module("saplings.api.retrieval")
            FaissVectorStore = retrieval_module.FaissVectorStore

            # Check for GPU configuration
            use_gpu = getattr(config.vector_store, "use_gpu", False)
            return FaissVectorStore(config, use_gpu=use_gpu)
        except (ImportError, AttributeError):
            # Fall back to in-memory if FAISS is unavailable
            return InMemoryVectorStore(config)
    if store_type == VectorStoreType.IN_MEMORY:
        # Default to in-memory vector store
        return InMemoryVectorStore(config)
    # For other store types, delegate to the plugin-based resolver
    from saplings.memory._internal.vector_store import get_vector_store as plugin_resolver

    return plugin_resolver(config)
