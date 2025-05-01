"""
Memory module for Saplings.

This module provides the memory store functionality for Saplings, including:
- Vector store for efficient similarity search
- Dependency graph for representing relationships between documents
- Indexers for extracting entities and relationships
- SecureStore for privacy-preserving storage

The memory module is designed to be extensible, with pluggable backends for
vector storage and indexing.
"""

from saplings.memory.config import MemoryConfig
from saplings.memory.document import Document, DocumentMetadata
from saplings.memory.graph import DependencyGraph, DocumentNode, EntityNode, Relationship
from saplings.memory.indexer import Indexer
from saplings.memory.memory_store import MemoryStore
from saplings.memory.vector_store import InMemoryVectorStore, VectorStore, get_vector_store

__all__ = [
    "MemoryConfig",
    "Document",
    "DocumentMetadata",
    "DependencyGraph",
    "EntityNode",
    "DocumentNode",
    "Relationship",
    "Indexer",
    "MemoryStore",
    "VectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
]
