from __future__ import annotations

"""
In-memory vector store implementation for Saplings memory.

This module provides an in-memory implementation of the VectorStore interface.
"""


import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from saplings.memory._internal.config import MemoryConfig, SimilarityMetric
from saplings.memory._internal.vector_store.vector_store import VectorStore

if TYPE_CHECKING:
    import builtins

    from saplings.memory._internal.document import Document

logger = logging.getLogger(__name__)


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
        self.documents: dict[str, "Document"] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.similarity_metric = self.config.vector_store.similarity_metric

    def add_document(self, document: "Document") -> None:
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
        if hasattr(document, "chunks"):
            for chunk in document.chunks:
                if chunk.embedding is not None:
                    self.documents[chunk.id] = chunk
                    self.embeddings[chunk.id] = chunk.embedding

    def add_documents(self, documents: builtins.list["Document"]) -> None:
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
    ) -> builtins.list[tuple["Document", float]]:
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

        # Sort by score (higher is better)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        results = [(self.documents[doc_id], score) for doc_id, score in scores[:limit]]

        return results

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate similarity between two vectors.

        Args:
        ----
            vec1: First vector
            vec2: Second vector

        Returns:
        -------
            float: Similarity score

        """
        if self.similarity_metric == SimilarityMetric.COSINE:
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
            # Dot product
            return float(np.dot(vec1, vec2))
        elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance (converted to similarity)
            dist = np.linalg.norm(vec1 - vec2)
            return float(1.0 / (1.0 + dist))
        else:
            # Default to cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _matches_filter(self, document: "Document", filter_dict: dict[str, Any]) -> bool:
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
            # Check metadata
            if hasattr(document, "metadata") and document.metadata:
                # Check custom metadata
                if hasattr(document.metadata, "custom") and document.metadata.custom:
                    if key in document.metadata.custom and document.metadata.custom[key] != value:
                        return False

                # Check regular metadata
                if hasattr(document.metadata, key):
                    if getattr(document.metadata, key) != value:
                        return False

            # Check document attributes
            if hasattr(document, key):
                if getattr(document, key) != value:
                    return False

        return True

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

    def update(self, document: "Document") -> None:
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

    def get(self, document_id: str) -> "Document | None":
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
    ) -> builtins.list["Document"]:
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
        documents_data = {}
        for doc_id, doc in self.documents.items():
            try:
                documents_data[doc_id] = doc.to_dict()
            except Exception as e:
                logger.warning(f"Failed to serialize document {doc_id}: {e}")

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

        logger.info(f"Saved vector store to {directory}")

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

                # Import here to avoid circular imports
                from saplings.memory._internal.document import Document

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

        logger.info(f"Loaded vector store from {directory}")
