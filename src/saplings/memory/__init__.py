from __future__ import annotations

"""
Memory module for Saplings.

This module re-exports the public API from saplings.api.memory.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.memory.

# Define the list of exported symbols
__all__ = [
    "DependencyGraph",
    "DependencyGraphBuilder",
    "Document",
    "DocumentMetadata",
    "DocumentNode",
    "InMemoryVectorStore",
    "Indexer",
    "IndexerRegistry",
    "MemoryConfig",
    "MemoryStore",
    "MemoryStoreBuilder",
    "SimpleIndexer",
    "get_indexer",
    "get_vector_store",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        # Handle DocumentNode separately
        if name == "DocumentNode":
            from saplings.api.document_node import DocumentNode

            return DocumentNode

        # Handle InMemoryVectorStore and get_vector_store separately
        if name == "InMemoryVectorStore":
            return InMemoryVectorStore
        if name == "get_vector_store":
            return get_vector_store

        # Import from the public API for all other names
        from saplings.api.memory import (
            DependencyGraph,
            DependencyGraphBuilder,
            Document,
            DocumentMetadata,
            Indexer,
            IndexerRegistry,
            MemoryConfig,
            MemoryStore,
            MemoryStoreBuilder,
            SimpleIndexer,
            get_indexer,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "DependencyGraph": DependencyGraph,
            "DependencyGraphBuilder": DependencyGraphBuilder,
            "Document": Document,
            "DocumentMetadata": DocumentMetadata,
            "Indexer": Indexer,
            "IndexerRegistry": IndexerRegistry,
            "MemoryConfig": MemoryConfig,
            "MemoryStore": MemoryStore,
            "MemoryStoreBuilder": MemoryStoreBuilder,
            "SimpleIndexer": SimpleIndexer,
            "get_indexer": get_indexer,
        }

        # Return the requested attribute
        return globals_dict.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# These are not in the current API, but we'll keep them for backward compatibility
# We'll use a simple class that inherits from VectorStore for backward compatibility
# Create a simple concrete implementation of VectorStore
class InMemoryVectorStore:
    """Simple in-memory vector store for backward compatibility."""

    def __init__(self):
        self._documents = {}

    def add_document(self, document_id, embedding, metadata=None):
        """Add a document to the store."""
        self._documents[document_id] = (embedding, metadata or {})

    def add_documents(self, documents):
        """Add multiple documents to the store."""
        for doc_id, embedding, metadata in documents:
            self.add_document(doc_id, embedding, metadata)

    def search(self, query_embedding, limit=10):
        """Search for documents similar to the query embedding."""
        return []

    def get_document(self, document_id):
        """Get a document by ID."""
        return self._documents.get(document_id)

    def get_documents(self, document_ids=None):
        """Get documents by IDs."""
        if document_ids is None:
            return list(self._documents.items())
        return [
            (doc_id, *self._documents[doc_id])
            for doc_id in document_ids
            if doc_id in self._documents
        ]

    def delete_document(self, document_id):
        """Delete a document by ID."""
        if document_id in self._documents:
            del self._documents[document_id]

    def delete_documents(self, document_ids):
        """Delete multiple documents by IDs."""
        for doc_id in document_ids:
            self.delete_document(doc_id)

    def clear(self):
        """Clear all documents from the store."""
        self._documents.clear()

    def count(self):
        """Count the number of documents in the store."""
        return len(self._documents)

    def save(self, path):
        """Save the store to a file."""

    def load(self, path):
        """Load the store from a file."""

    def delete(self, document_id):
        """Delete a document by ID (alias for delete_document)."""
        self.delete_document(document_id)

    def update(self, document_id, embedding, metadata=None):
        """Update a document in the store."""
        self.add_document(document_id, embedding, metadata)


def get_vector_store():
    """Get a vector store instance."""
    return InMemoryVectorStore()
