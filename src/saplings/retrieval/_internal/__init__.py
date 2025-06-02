from __future__ import annotations

"""
Internal module for retrieval components.

This module provides the implementation of retrieval components for the Saplings framework.
"""

# Import from individual modules
from saplings.retrieval._internal.cache_config import CacheConfig
from saplings.retrieval._internal.config import RetrievalConfig
from saplings.retrieval._internal.expansion import (
    EntropyCalculator,
    GraphExpander,
)

# Import from subdirectories
from saplings.retrieval._internal.retrievers import (
    CachedEmbeddingRetriever,
    CascadeRetriever,
    EmbeddingRetriever,
    TFIDFRetriever,
)
from saplings.retrieval._internal.vector_store import (
    FaissVectorStore,
)

__all__ = [
    # Configuration
    "CacheConfig",
    "RetrievalConfig",
    # Retrievers
    "CachedEmbeddingRetriever",
    "CascadeRetriever",
    "EmbeddingRetriever",
    "TFIDFRetriever",
    # Expansion
    "EntropyCalculator",
    "GraphExpander",
    # Vector store
    "FaissVectorStore",
]
