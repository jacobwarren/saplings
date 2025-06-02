from __future__ import annotations

"""
Retrievers module for retrieval components.

This module provides retriever implementations for the Saplings framework.
"""

from saplings.retrieval._internal.retrievers.cached_embedding_retriever import (
    CachedEmbeddingRetriever,
)
from saplings.retrieval._internal.retrievers.cascade_retriever import CascadeRetriever
from saplings.retrieval._internal.retrievers.embedding_retriever import EmbeddingRetriever
from saplings.retrieval._internal.retrievers.tfidf_retriever import TFIDFRetriever

__all__ = [
    "CachedEmbeddingRetriever",
    "CascadeRetriever",
    "EmbeddingRetriever",
    "TFIDFRetriever",
]
