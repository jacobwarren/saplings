from __future__ import annotations

"""
Container initialization hooks for Saplings.

This module defines hooks that are called during container initialization to register
additional components or modify the container configuration.
"""


from saplings.agent_config import AgentConfig
from saplings.di import container
from saplings.memory.config import MemoryConfig, VectorStoreType
from saplings.memory.vector_store import VectorStore


def register_vector_store(config: AgentConfig | None = None) -> None:
    """
    Register the appropriate vector store based on configuration.

    Args:
    ----
        config: Optional agent configuration

    """
    config = config or AgentConfig(provider="test", model_name="model")
    memory_config = MemoryConfig(chunk_size=1000, chunk_overlap=200)

    # Check if FAISS is enabled in configuration
    use_faiss = memory_config.vector_store.store_type == VectorStoreType.FAISS

    if use_faiss:
        # Use FAISS if configured
        try:
            from saplings.retrieval.faiss_vector_store import FaissVectorStore

            container.register(
                VectorStore,
                factory=lambda: FaissVectorStore(
                    memory_config, use_gpu=memory_config.vector_store.use_gpu
                ),
            )
        except ImportError:
            # Fall back to in-memory store if FAISS is not available
            from saplings.memory.vector_store import InMemoryVectorStore

            container.register(VectorStore, factory=lambda: InMemoryVectorStore(memory_config))
    else:
        # Use in-memory store by default
        from saplings.memory.vector_store import InMemoryVectorStore

        container.register(VectorStore, factory=lambda: InMemoryVectorStore(memory_config))


def register_optimized_packers():
    """Register optimized block packers based on available backends."""
    # The block_pack module automatically selects the best implementation
    # We don't need explicit registration here as it's handled transparently


def initialize_hooks(config: AgentConfig | None = None) -> None:
    """
    Initialize all container hooks.

    Args:
    ----
        config: Optional agent configuration

    """
    register_vector_store(config)
    register_optimized_packers()
