from __future__ import annotations

"""
Retrieval API module for Saplings.

This module provides the public API for retrieval components.
"""

from saplings.api.stability import beta, stable
from saplings.retrieval._internal import (
    CachedEmbeddingRetriever as _CachedEmbeddingRetriever,
)
from saplings.retrieval._internal import (
    CascadeRetriever as _CascadeRetriever,
)
from saplings.retrieval._internal import (
    EmbeddingRetriever as _EmbeddingRetriever,
)
from saplings.retrieval._internal import (
    EntropyCalculator as _EntropyCalculator,
)
from saplings.retrieval._internal import (
    FaissVectorStore as _FaissVectorStore,
)
from saplings.retrieval._internal import (
    GraphExpander as _GraphExpander,
)
from saplings.retrieval._internal import (
    RetrievalConfig as _RetrievalConfig,
)
from saplings.retrieval._internal import (
    TFIDFRetriever as _TFIDFRetriever,
)
from saplings.retrieval._internal.config import EntropyConfig as _EntropyConfig


@beta
class CascadeRetriever(_CascadeRetriever):
    """
    Orchestrates the retrieval pipeline with multiple stages.

    This retriever combines multiple retrieval strategies in a cascade:
    1. TF-IDF initial filtering
    2. Embedding-based similarity search
    3. Graph expansion using the dependency graph
    4. Entropy-based termination logic

    The cascade approach progressively refines results for better relevance.
    """


@beta
class RetrievalConfig(_RetrievalConfig):
    """
    Configuration for retrieval components.

    This class defines the configuration options for retrieval components,
    including parameters for TF-IDF, embedding, graph expansion, and entropy.
    """


@beta
class EntropyConfig(_EntropyConfig):
    """
    Configuration for entropy-based termination.

    This class defines the configuration options for entropy-based termination
    logic in the retrieval pipeline.
    """


@beta
class EmbeddingRetriever(_EmbeddingRetriever):
    """
    Retriever that uses embeddings for similarity search.

    This retriever uses vector embeddings to find semantically similar documents
    based on cosine similarity or other distance metrics.
    """


@beta
class EntropyCalculator(_EntropyCalculator):
    """
    Calculator for determining when to stop retrieval.

    This component calculates the entropy of retrieval results to determine
    when additional retrieval would not add significant information.
    """


@beta
class GraphExpander(_GraphExpander):
    """
    Expander that uses the dependency graph to find related documents.

    This component expands retrieval results by following edges in the
    dependency graph to find documents that are related to the initial results.
    """


@beta
class TFIDFRetriever(_TFIDFRetriever):
    """
    Retriever that uses TF-IDF for initial filtering.

    This retriever uses term frequency-inverse document frequency (TF-IDF)
    to quickly filter documents based on keyword matching before more
    expensive semantic search.
    """


@beta
class CachedEmbeddingRetriever(_CachedEmbeddingRetriever):
    """
    Cached embedding retriever for semantic search.

    This retriever extends the standard EmbeddingRetriever with caching capabilities
    using the unified caching system. It caches both document embeddings and
    retrieval results for improved performance.

    Features:
    - Caches document embeddings to avoid recomputing them
    - Caches retrieval results for identical queries
    - Configurable cache TTL, provider, and eviction strategy
    """


@stable
class FaissVectorStore(_FaissVectorStore):
    """
    Vector store implementation using FAISS.

    This vector store uses Facebook AI Similarity Search (FAISS) for efficient
    similarity search in high-dimensional spaces. It supports both CPU and GPU
    acceleration for fast retrieval of similar vectors.
    """


__all__ = [
    # Configuration
    "RetrievalConfig",
    "EntropyConfig",
    # Retrievers
    "CascadeRetriever",
    "EmbeddingRetriever",
    "TFIDFRetriever",
    "CachedEmbeddingRetriever",
    # Components
    "EntropyCalculator",
    "GraphExpander",
    "FaissVectorStore",
]
