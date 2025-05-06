from __future__ import annotations

"""
Retrieval service interface for Saplings.

This module defines the interface for retrieval operations that find relevant
documents based on queries. This is a pure interface with no implementation
details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any

# Forward references
Document = Any  # From saplings.memory.document


class IRetrievalService(ABC):
    """Interface for retrieval operations."""

    @abstractmethod
    async def retrieve(
        self, query: str, limit: int | None = None, timeout: float | None = None
    ) -> list[Document]:
        """
        Retrieve documents based on a query.

        Args:
        ----
            query: The search query
            limit: Maximum number of documents to return
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Document]: Retrieved documents

        """

    @abstractmethod
    async def retrieve_with_scores(
        self, query: str, limit: int | None = None, timeout: float | None = None
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents with relevance scores.

        Args:
        ----
            query: The search query
            limit: Maximum number of documents to return
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Tuple[Document, float]]: Documents with scores

        """

    @abstractmethod
    async def retrieve_by_ids(
        self, document_ids: list[str], timeout: float | None = None
    ) -> list[Document]:
        """
        Retrieve documents by their IDs.

        Args:
        ----
            document_ids: List of document IDs
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Document]: Retrieved documents

        """

    @abstractmethod
    async def calculate_entropy(
        self, query: str, documents: list[Document], timeout: float | None = None
    ) -> float:
        """
        Calculate entropy for a query against documents.

        Args:
        ----
            query: The search query
            documents: List of documents to calculate entropy for
            timeout: Optional timeout in seconds

        Returns:
        -------
            float: Entropy score

        """
