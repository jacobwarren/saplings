from __future__ import annotations

"""
Vector store module for Saplings memory.

This module defines the VectorStore abstract base class and implementations.
"""


import datetime
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from saplings.memory._internal.config import (
    MemoryConfig,
    PrivacyLevel,
    SimilarityMetric,
    VectorStoreType,
)
from saplings.memory._internal.document import Document

if TYPE_CHECKING:
    import builtins

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """
    Abstract base class for vector stores.

    A vector store is responsible for storing document embeddings and performing
    similarity searches.
    """

    @abstractmethod
    def add_document(self, document: Document) -> None:
        """
        Add a document to the vector store.

        Args:
        ----
            document: Document to add

        """

    @abstractmethod
    def add_documents(self, documents: builtins.list[Document]) -> None:
        """
        Add multiple documents to the vector store.

        Args:
        ----
            documents: Documents to add

        """

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> builtins.list[tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
        ----
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_dict: Optional filter criteria

        Returns:
        -------
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples

        """

    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
        ----
            document_id: ID of the document to delete

        Returns:
        -------
            bool: True if the document was deleted, False otherwise

        """

    @abstractmethod
    def update(self, document: Document) -> None:
        """
        Update a document in the vector store.

        Args:
        ----
            document: Updated document

        """

    @abstractmethod
    def get(self, document_id: str) -> Document | None:
        """
        Get a document by ID.

        Args:
        ----
            document_id: ID of the document to get

        Returns:
        -------
            Optional[Document]: Document if found, None otherwise

        """

    @abstractmethod
    def list(
        self, limit: int = 100, filter_dict: dict[str, Any] | None = None
    ) -> builtins.list[Document]:
        """
        List documents in the vector store.

        Args:
        ----
            limit: Maximum number of documents to return
            filter_dict: Optional filter criteria

        Returns:
        -------
            List[Document]: List of documents

        """

    @abstractmethod
    def count(self, filter_dict: dict[str, Any] | None = None) -> int:
        """
        Count documents in the vector store.

        Args:
        ----
            filter_dict: Optional filter criteria

        Returns:
        -------
            int: Number of documents

        """

    @abstractmethod
    def clear(self):
        """Clear all documents from the vector store."""

    @abstractmethod
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.

        Args:
        ----
            directory: Directory to save to

        """

    @abstractmethod
    def load(self, directory: str) -> None:
        """
        Load the vector store from disk.

        Args:
        ----
            directory: Directory to load from

        """


class InMemoryVectorStore(VectorStore):
    """
    In-memory implementation of the VectorStore interface.

    This implementation stores documents and embeddings in memory using numpy arrays.
    It's suitable for small to medium-sized collections but doesn't scale to very large datasets.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """
        Initialize the in-memory vector store.

        Args:
        ----
            config: Memory configuration

        """
        self.config = config or MemoryConfig.default()
        self.documents: dict[str, Document] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.similarity_metric = self.config.vector_store.similarity_metric

    def add_document(self, document: Document) -> None:
        """
        Add a document to the vector store.

        Args:
        ----
            document: Document to add

        """
        if document.embedding is None:
            msg = f"Document {document.id} has no embedding"
            raise ValueError(msg)

        self.documents[document.id] = document
        self.embeddings[document.id] = document.embedding

        # Also add any chunks
        for chunk in document.chunks:
            if chunk.embedding is not None:
                self.documents[chunk.id] = chunk
                self.embeddings[chunk.id] = chunk.embedding

    def add_documents(self, documents: builtins.list[Document]) -> None:
        """
        Add multiple documents to the vector store.

        Args:
        ----
            documents: Documents to add

        """
        for document in documents:
            self.add_document(document)

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
    ) -> builtins.list[tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
        ----
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_dict: Optional filter criteria

        Returns:
        -------
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples

        """
        if not self.embeddings:
            return []

        # Filter documents if filter_dict is provided
        doc_ids = list(self.embeddings.keys())
        if filter_dict:
            doc_ids = [
                doc_id
                for doc_id in doc_ids
                if self._matches_filter(self.documents[doc_id], filter_dict)
            ]

        if not doc_ids:
            return []

        # Calculate similarity scores
        scores = []
        for doc_id in doc_ids:
            embedding = self.embeddings[doc_id]
            score = self._calculate_similarity(query_embedding, embedding)
            scores.append((doc_id, score))

        # Sort by score (descending) and take top 'limit'
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:limit]

        # Return documents with scores
        return [(self.documents[doc_id], score) for doc_id, score in top_scores]

    def delete(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
        ----
            document_id: ID of the document to delete

        Returns:
        -------
            bool: True if the document was deleted, False otherwise

        """
        if document_id in self.documents:
            del self.documents[document_id]
            del self.embeddings[document_id]
            return True
        return False

    def update(self, document: Document) -> None:
        """
        Update a document in the vector store.

        Args:
        ----
            document: Updated document

        """
        if document.id not in self.documents:
            msg = f"Document {document.id} not found"
            raise ValueError(msg)

        if document.embedding is None:
            msg = f"Document {document.id} has no embedding"
            raise ValueError(msg)

        self.documents[document.id] = document
        self.embeddings[document.id] = document.embedding

    def get(self, document_id: str) -> Document | None:
        """
        Get a document by ID.

        Args:
        ----
            document_id: ID of the document to get

        Returns:
        -------
            Optional[Document]: Document if found, None otherwise

        """
        return self.documents.get(document_id)

    def list(
        self, limit: int = 100, filter_dict: dict[str, Any] | None = None
    ) -> builtins.list[Document]:
        """
        List documents in the vector store.

        Args:
        ----
            limit: Maximum number of documents to return
            filter_dict: Optional filter criteria

        Returns:
        -------
            List[Document]: List of documents

        """
        if filter_dict:
            docs = [
                doc for doc in self.documents.values() if self._matches_filter(doc, filter_dict)
            ]
        else:
            docs = list(self.documents.values())

        return docs[:limit]

    def count(self, filter_dict: dict[str, Any] | None = None) -> int:
        """
        Count documents in the vector store.

        Args:
        ----
            filter_dict: Optional filter criteria

        Returns:
        -------
            int: Number of documents

        """
        if filter_dict:
            return len(
                [doc for doc in self.documents.values() if self._matches_filter(doc, filter_dict)]
            )
        return len(self.documents)

    def clear(self):
        """Clear all documents from the vector store."""
        self.documents.clear()
        self.embeddings.clear()

    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.

        Args:
        ----
            directory: Directory to save to

        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save documents
        documents_data = {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}

        # Custom JSON encoder to handle datetime objects
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime.datetime):
                    return obj.isoformat()
                return super().default(obj)

        with open(directory_path / "documents.json", "w") as f:
            json.dump(documents_data, f, cls=DateTimeEncoder)

        # Save embeddings
        embeddings_data = {
            doc_id: embedding.tolist() for doc_id, embedding in self.embeddings.items()
        }
        with open(directory_path / "embeddings.json", "w") as f:
            json.dump(embeddings_data, f)

        # Save config
        with open(directory_path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f)

    def load(self, directory: str) -> None:
        """
        Load the vector store from disk.

        Args:
        ----
            directory: Directory to load from

        """
        directory_path = Path(directory)

        # Load config
        config_path = directory_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                self.config = MemoryConfig(**config_data)

        # Load documents
        documents_path = directory_path / "documents.json"
        if documents_path.exists():
            with open(documents_path) as f:
                documents_data = json.load(f)
                self.documents = {
                    doc_id: Document.from_dict(doc_data)
                    for doc_id, doc_data in documents_data.items()
                }

        # Load embeddings
        embeddings_path = directory_path / "embeddings.json"
        if embeddings_path.exists():
            with open(embeddings_path) as f:
                embeddings_data = json.load(f)
                self.embeddings = {
                    doc_id: np.array(embedding, dtype=np.float32)
                    for doc_id, embedding in embeddings_data.items()
                }

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings.

        Args:
        ----
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
        -------
            float: Similarity score

        """
        if self.similarity_metric == SimilarityMetric.COSINE:
            # Cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

        if self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
            # Dot product
            return float(np.dot(embedding1, embedding2))

        if self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert distance to similarity (1 / (1 + distance)) and ensure it's a Python float
            return float(1.0 / (1.0 + distance))

        msg = f"Unknown similarity metric: {self.similarity_metric}"
        raise ValueError(msg)

    def _matches_filter(self, document: Document, filter_dict: dict[str, Any]) -> bool:
        """
        Check if a document matches a filter.

        Args:
        ----
            document: Document to check
            filter_dict: Filter criteria

        Returns:
        -------
            bool: True if the document matches the filter, False otherwise

        """
        for key, value in filter_dict.items():
            # Check document ID
            if key == "id" and document.id != value:
                return False

            # Check metadata fields
            if key.startswith("metadata."):
                field = key[len("metadata.") :]

                # Handle nested fields with dot notation
                if "." in field:
                    parts = field.split(".")
                    obj = document.metadata
                    for part in parts[:-1]:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        elif isinstance(obj, dict) and part in obj:
                            obj = obj[part]
                        else:
                            return False

                    last_part = parts[-1]
                    if hasattr(obj, last_part):
                        field_value = getattr(obj, last_part)
                    elif isinstance(obj, dict) and last_part in obj:
                        field_value = obj[last_part]
                    else:
                        return False

                # Handle direct metadata fields
                elif hasattr(document.metadata, field):
                    field_value = getattr(document.metadata, field)
                elif hasattr(document.metadata, "custom") and document.metadata is not None:
                    custom = getattr(document.metadata, "custom", None)
                    if isinstance(custom, dict) and field in custom:
                        field_value = custom[field]
                    else:
                        return False
                else:
                    return False

                # Check if the field value matches the filter value
                if isinstance(value, list):
                    if field_value not in value:
                        return False
                elif field_value != value:
                    return False

            # Check content (substring match)
            elif key == "content" and value not in document.content:
                return False

        return True


def get_vector_store(config: MemoryConfig | None = None, *, registry=None) -> VectorStore:
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

        # Look for a plugin with a matching name
        for plugin_name, plugin_class in memory_store_plugins.items():
            if plugin_name.lower() == store_type.lower() or plugin_name.lower().replace(
                "_", ""
            ) == store_type.lower().replace("_", ""):
                # Check if the plugin class is a VectorStore
                if issubclass(plugin_class, VectorStore):
                    # Create an instance of the plugin
                    # Since we can't pass config directly due to type issues,
                    # we'll create a wrapper class that inherits from the plugin class
                    # and overrides the __init__ method to accept config

                    class ConfigurableVectorStore(plugin_class):  # type: ignore
                        """A wrapper class that allows passing config to a plugin."""

                        def __init__(self, config: MemoryConfig | None = None) -> None:
                            """Initialize with config."""
                            # Call the parent class's __init__ without arguments
                            super().__init__()
                            # Set the config attribute if it exists
                            if hasattr(self, "config"):
                                self.config = config

                    return ConfigurableVectorStore(config)
                logger.warning(f"Plugin {plugin_name} is not a VectorStore")
                # Fall back to in-memory store
                return InMemoryVectorStore(config)

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
                        # Since we can't pass config directly due to type issues,
                        # we'll create a wrapper class that inherits from the plugin class
                        # and overrides the __init__ method to accept config

                        class ConfigurableVectorStore(plugin_class):  # type: ignore
                            """A wrapper class that allows passing config to a plugin."""

                            def __init__(self, config: MemoryConfig | None = None) -> None:
                                """Initialize with config."""
                                # Call the parent class's __init__ without arguments
                                super().__init__()
                                # Set the config attribute if it exists
                                if hasattr(self, "config"):
                                    self.config = config

                        return ConfigurableVectorStore(config)
                    logger.warning(f"Plugin {plugin_name} is not a VectorStore")
                    # Fall back to in-memory store
                    return InMemoryVectorStore(config)

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
