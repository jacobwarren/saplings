from __future__ import annotations

"""
Retrieval module for Saplings.

This module provides the retrieval functionality for Saplings, including:
- TF-IDF initial filtering
- Embedding-based similarity search
- Graph expansion using the dependency graph
- Entropy-based termination logic
- CascadeRetriever for orchestrating the retrieval pipeline

The retrieval module is designed to be efficient and context-aware,
with a cascaded approach that progressively refines results.
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.retrieval.

__all__ = [
    "CachedEmbeddingRetriever",
    "CascadeRetriever",
    "EmbeddingRetriever",
    "EntropyCalculator",
    "FaissVectorStore",
    "GraphExpander",
    "RetrievalConfig",
    "TFIDFRetriever",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        # Import from the public API
        from saplings.api.retrieval import (
            CachedEmbeddingRetriever,
            CascadeRetriever,
            EmbeddingRetriever,
            EntropyCalculator,
            FaissVectorStore,
            GraphExpander,
            RetrievalConfig,
            TFIDFRetriever,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "CachedEmbeddingRetriever": CachedEmbeddingRetriever,
            "CascadeRetriever": CascadeRetriever,
            "EmbeddingRetriever": EmbeddingRetriever,
            "EntropyCalculator": EntropyCalculator,
            "FaissVectorStore": FaissVectorStore,
            "GraphExpander": GraphExpander,
            "RetrievalConfig": RetrievalConfig,
            "TFIDFRetriever": TFIDFRetriever,
        }

        # Return the requested attribute
        return globals_dict.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
