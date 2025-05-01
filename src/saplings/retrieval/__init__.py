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

from saplings.retrieval.cascade_retriever import CascadeRetriever
from saplings.retrieval.config import RetrievalConfig
from saplings.retrieval.embedding_retriever import EmbeddingRetriever
from saplings.retrieval.entropy_calculator import EntropyCalculator
from saplings.retrieval.graph_expander import GraphExpander
from saplings.retrieval.tfidf_retriever import TFIDFRetriever

__all__ = [
    "RetrievalConfig",
    "TFIDFRetriever",
    "EmbeddingRetriever",
    "GraphExpander",
    "EntropyCalculator",
    "CascadeRetriever",
]
