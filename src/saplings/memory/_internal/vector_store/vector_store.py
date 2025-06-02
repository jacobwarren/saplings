from __future__ import annotations

"""
Vector store module for Saplings memory.

This module defines the VectorStore abstract base class and implementations.
"""


import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import builtins

    from saplings.memory._internal.document import Document

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """
    Abstract base class for vector stores.

    A vector store is responsible for storing document embeddings and performing
    similarity searches.
    """

    @abstractmethod
    def add_document(self, document: "Document") -> None:
        """
        Add a document to the vector store.

        Args:
        ----
            document: Document to add

        """

    @abstractmethod
    def add_documents(self, documents: builtins.list["Document"]) -> None:
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
    def update(self, document: "Document") -> None:
        """
        Update a document in the vector store.

        Args:
        ----
            document: Updated document

        """

    @abstractmethod
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

    @abstractmethod
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
