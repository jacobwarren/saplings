from __future__ import annotations

"""
Vector store factory module for Saplings memory.

This module provides a factory function for creating vector store instances.
"""

import logging
from typing import TYPE_CHECKING

from saplings.memory._internal.config import MemoryConfig, PrivacyLevel, VectorStoreType
from saplings.memory._internal.vector_store.in_memory_vector_store import InMemoryVectorStore
from saplings.memory._internal.vector_store.vector_store import VectorStore

if TYPE_CHECKING:
    from saplings.core._internal.plugin import PluginRegistry

logger = logging.getLogger(__name__)


def get_vector_store(
    config: MemoryConfig | None = None, *, registry: "PluginRegistry | None" = None
) -> VectorStore:
    """
    Get a vector store instance based on configuration.

    Args:
    ----
        config: Memory configuration
        registry: Optional plugin registry to use

    Returns:
    -------
        VectorStore: Vector store instance

    """
    config = config or MemoryConfig.default()
    store_type = config.vector_store.store_type

    # Check for built-in vector store types
    if store_type == VectorStoreType.IN_MEMORY:
        return InMemoryVectorStore(config)

    # Check for plugin-based vector stores
    try:
        from saplings.core._internal.plugin import PluginType, get_plugins_by_type

        # Get all memory store plugins
        memory_store_plugins = get_plugins_by_type(PluginType.MEMORY_STORE, registry=registry)

        # If we're looking for a specific store type, try to find a matching plugin
        if store_type == VectorStoreType.CUSTOM:
            store_name = config.vector_store.custom_store_name
            if store_name and store_name in memory_store_plugins:
                plugin_class = memory_store_plugins[store_name]
                if issubclass(plugin_class, VectorStore):
                    return plugin_class(config)

        # If we're looking for a secure store, try to find a secure memory store plugin
        if (
            store_type == VectorStoreType.CUSTOM
            and config.secure_store.privacy_level != PrivacyLevel.NONE
        ):
            for plugin_name, plugin_class in memory_store_plugins.items():
                if "secure" in plugin_name.lower():
                    # Check if the plugin class is a VectorStore
                    if issubclass(plugin_class, VectorStore):
                        # Create an instance of the plugin
                        return plugin_class(config)

    except ImportError:
        # Plugin system not available
        pass

    # Add support for other vector store types here
    # elif store_type == VectorStoreType.FAISS:
    #     return FaissVectorStore(config)
    # elif store_type == VectorStoreType.QDRANT:
    #     return QdrantVectorStore(config)
    # elif store_type == VectorStoreType.PINECONE:
    #     return PineconeVectorStore(config)

    msg = f"Unsupported vector store type: {store_type}"
    raise ValueError(msg)
