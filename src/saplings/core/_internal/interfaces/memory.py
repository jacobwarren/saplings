from __future__ import annotations

"""
Memory manager interface for Saplings.

This module defines the interface for memory management operations that handle
documents and their storage. This is a pure interface with no implementation
details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Forward references
Document = Any  # From saplings.memory.document


@dataclass
class MemoryConfig:
    """Configuration for memory operations."""

    store_type: str = "vector"
    vector_dimensions: int = 1536
    similarity_threshold: float = 0.7
    max_documents: int = 100
    ttl: Optional[int] = None


@dataclass
class MemoryResult:
    """Result of a memory operation."""

    success: bool
    document_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


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

    @abstractmethod
    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[MemoryConfig] = None,
        trace_id: Optional[str] = None,
    ) -> MemoryResult:
        """
        Store content in memory.

        Args:
        ----
            content: Content to store
            metadata: Optional metadata
            config: Optional memory configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            MemoryResult: Result of the store operation

        """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        config: Optional[MemoryConfig] = None,
        trace_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve documents from memory based on query.

        Args:
        ----
            query: Query string
            limit: Maximum number of documents to retrieve
            config: Optional memory configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            List[Document]: Retrieved documents

        """
