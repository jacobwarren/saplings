from __future__ import annotations

"""
Document Node API module for Saplings.

This module provides the public API for document node functionality.
"""

from saplings.api.memory.document import Document
from saplings.api.stability import beta


@beta
class DocumentNode:
    """
    Node in a dependency graph representing a document.

    This class represents a document node in a dependency graph, with
    properties for the document ID, content, and metadata.
    """

    def __init__(self, document: Document, metadata: dict | None = None) -> None:
        """
        Initialize a document node.

        Args:
        ----
            document: Document
            metadata: Additional metadata

        """
        self.id = document.id
        self.document = document
        self.metadata = metadata or {}

    def __eq__(self, other: object) -> bool:
        """
        Check if two nodes are equal.

        Args:
        ----
            other: Other node

        Returns:
        -------
            bool: True if the nodes are equal, False otherwise

        """
        if not isinstance(other, DocumentNode):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """
        Get the hash of the node.

        Returns
        -------
            int: Hash value

        """
        return hash(self.id)

    def to_dict(self) -> dict:
        """
        Convert the document node to a dictionary.

        Returns
        -------
            dict: Dictionary representation

        """
        data = {"id": self.id, "metadata": self.metadata, "document_id": self.document.id}
        return data


__all__ = ["DocumentNode"]
