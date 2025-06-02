from __future__ import annotations

"""
Internal module for memory components.

This module provides the implementation of memory components for the Saplings framework.
"""

# Import from config module
from saplings.memory._internal.config import MemoryConfig

# Import from document module
from saplings.memory._internal.document import Document, DocumentMetadata

# Import from graph module
from saplings.memory._internal.graph.dependency_graph import DependencyGraph
from saplings.memory._internal.graph.dependency_graph_builder import DependencyGraphBuilder
from saplings.memory._internal.graph.relationship import Relationship

# Import from indexing module
from saplings.memory._internal.indexing.entity import Entity
from saplings.memory._internal.indexing.indexer import Indexer
from saplings.memory._internal.indexing.indexer_registry import (
    IndexerRegistry,
    get_indexer,
    get_indexer_registry,
)
from saplings.memory._internal.indexing.indexing_result import IndexingResult
from saplings.memory._internal.indexing.simple_indexer import SimpleIndexer

# Import memory store components
from saplings.memory._internal.memory_store import MemoryStore
from saplings.memory._internal.memory_store_builder import MemoryStoreBuilder

# Import from paper module
from saplings.memory._internal.paper.paper_chunker import build_section_relationships, chunk_paper
from saplings.memory._internal.paper.paper_processor import process_paper
from saplings.memory._internal.vector_store.get_vector_store import get_vector_store
from saplings.memory._internal.vector_store.in_memory_vector_store import InMemoryVectorStore

# Import from vector_store module
from saplings.memory._internal.vector_store.vector_store import VectorStore

__all__ = [
    # Config
    "MemoryConfig",
    # Document
    "Document",
    "DocumentMetadata",
    # Graph
    "DependencyGraph",
    "DependencyGraphBuilder",
    "Relationship",
    # Indexing
    "Entity",
    "Indexer",
    "IndexerRegistry",
    "IndexingResult",
    "SimpleIndexer",
    "get_indexer",
    "get_indexer_registry",
    # Paper
    "chunk_paper",
    "build_section_relationships",
    "process_paper",
    # Vector Store
    "VectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
    # Memory Store
    "MemoryStore",
    "MemoryStoreBuilder",
]
