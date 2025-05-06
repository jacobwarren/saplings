from __future__ import annotations

"""
Memory manager interface for Saplings.

This module defines the interface for memory management operations that handle
documents and their storage. This is a pure interface with no implementation
details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any, Callable

# Forward references
Document = Any  # From saplings.memory.document


class IMemoryManager(ABC):
    """Interface for memory management operations."""

    @abstractmethod
    async def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Document:
        """
        Add a document to memory.

        Args:
        ----
            content: Document content
            metadata: Optional metadata
            timeout: Optional timeout in seconds

        Returns:
        -------
            Document: The created document

        """

    @abstractmethod
    async def add_documents_from_directory(
        self, directory: str, extension: str = ".txt", timeout: float | None = None
    ) -> list[Document]:
        """
        Add documents from a directory.

        Args:
        ----
            directory: Directory path
            extension: File extension
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Document]: The created documents

        """

    @abstractmethod
    async def get_document(self, document_id: str, timeout: float | None = None) -> Document | None:
        """
        Get a document by ID.

        Args:
        ----
            document_id: Document ID
            timeout: Optional timeout in seconds

        Returns:
        -------
            Optional[Document]: The document if found

        """

    @abstractmethod
    async def get_documents(
        self,
        filter_func: Callable[[Document], bool] | None = None,
        timeout: float | None = None,
    ) -> list[Document]:
        """
        Get documents, optionally filtered.

        Args:
        ----
            filter_func: Optional filter function
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Document]: The matching documents

        """

    @property
    @abstractmethod
    def memory_store(self):
        """
        Get the underlying memory store.

        Returns
        -------
            Any: The memory store

        """

    @property
    @abstractmethod
    def dependency_graph(self):
        """
        Get the dependency graph.

        Returns
        -------
            Any: The dependency graph

        """
