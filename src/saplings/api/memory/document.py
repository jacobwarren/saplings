from __future__ import annotations

"""
Memory Document API module for Saplings.

This module provides the public API for memory documents.
"""

import uuid
from typing import Any, Dict, List, Optional, Union

from saplings.api.stability import stable


@stable
class DocumentMetadata:
    """
    Metadata for a document.

    This class represents metadata associated with a document, including
    source, content type, language, and other attributes.
    """

    def __init__(
        self,
        source: str = "",
        content_type: str = "text/plain",
        language: str = "en",
        author: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize document metadata.

        Args:
        ----
            source: Source of the document
            content_type: Content type of the document
            language: Language of the document
            author: Author of the document
            **kwargs: Additional metadata fields

        """
        self.source = source
        self.content_type = content_type
        self.language = language
        self.author = author
        self.additional = kwargs

    def __getitem__(self, key: str) -> Any:
        """
        Get a metadata field.

        Args:
        ----
            key: Field name

        Returns:
        -------
            Field value

        """
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a metadata field.

        Args:
        ----
            key: Field name
            value: Field value

        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.additional[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata field with a default value.

        Args:
        ----
            key: Field name
            default: Default value if field not found

        Returns:
        -------
            Field value or default

        """
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to a dictionary.

        Returns
        -------
            Dictionary representation

        """
        result = {
            "source": self.source,
            "content_type": self.content_type,
            "language": self.language,
            "author": self.author,
        }
        result.update(self.additional)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """
        Create metadata from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            DocumentMetadata instance

        """
        # Extract known fields
        source = data.pop("source", "")
        content_type = data.pop("content_type", "text/plain")
        language = data.pop("language", "en")
        author = data.pop("author", "")

        # Pass remaining fields as additional metadata
        return cls(
            source=source,
            content_type=content_type,
            language=language,
            author=author,
            **data,
        )


@stable
class Document:
    """
    Document for memory storage.

    This class represents a document that can be stored in memory.
    It contains the document content and metadata.
    """

    def __init__(
        self,
        content: str,
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
        id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ) -> None:
        """
        Initialize a document.

        Args:
        ----
            content: Document content
            metadata: Document metadata
            id: Document ID (generated if not provided)
            embedding: Document embedding vector

        """
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.embedding = embedding

        # Process metadata
        if metadata is None:
            self.metadata = DocumentMetadata()
        elif isinstance(metadata, dict):
            self.metadata = DocumentMetadata.from_dict(metadata)
        else:
            self.metadata = metadata

    def __str__(self) -> str:
        """
        Get string representation of the document.

        Returns
        -------
            String representation

        """
        return f"Document(id={self.id}, type={self.metadata.content_type}, source={self.metadata.source})"

    def __repr__(self) -> str:
        """
        Get string representation of the document.

        Returns
        -------
            String representation

        """
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document to a dictionary.

        Returns
        -------
            Dictionary representation

        """
        embedding_value = None
        if self.embedding is not None:
            embedding_value = (
                self.embedding.tolist() if hasattr(self.embedding, "tolist") else self.embedding
            )

        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "embedding": embedding_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create a document from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            Document instance

        """
        # Extract metadata
        metadata_dict = data.get("metadata", {})
        metadata = DocumentMetadata.from_dict(metadata_dict)

        # Extract embedding
        embedding = data.get("embedding")

        return cls(
            content=data["content"],
            metadata=metadata,
            id=data.get("id"),
            embedding=embedding,
        )

    @classmethod
    def create(
        cls,
        content: str,
        metadata: Optional[Union[Dict[str, Any], DocumentMetadata]] = None,
        embedding: Optional[List[float]] = None,
        id: Optional[str] = None,
    ) -> "Document":
        """
        Create a new document.

        Args:
        ----
            content: The content of the document
            metadata: Metadata for the document
            embedding: Embedding for the document
            id: ID for the document

        Returns:
        -------
            A new document

        """
        if isinstance(metadata, dict):
            metadata = DocumentMetadata.from_dict(metadata)

        # Convert to the expected types
        import numpy as np

        embedding_array = np.array(embedding) if embedding is not None else None
        doc_id = id or ""

        # Create default metadata if none provided
        if metadata is None:
            metadata = DocumentMetadata(
                source="",
                content_type="text/plain",
                language="en",
                author="",
            )

        return cls(
            content=content,
            metadata=metadata,
            embedding=embedding_array,
            id=doc_id,
        )

    def update_embedding(self, embedding: Any) -> None:
        """
        Update the document's embedding.

        Args:
        ----
            embedding: New embedding vector

        """
        self.embedding = embedding


__all__ = [
    "Document",
    "DocumentMetadata",
]
