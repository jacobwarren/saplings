"""
Document module for Saplings memory.

This module defines the Document class, which represents a document in the memory store.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    
    source: str = Field(..., description="Source of the document (e.g., file path, URL)")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    content_type: str = Field("text/plain", description="Content type (e.g., text/plain, text/markdown)")
    language: Optional[str] = Field(None, description="Language of the document")
    author: Optional[str] = Field(None, description="Author of the document")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the document")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
    
    def update(self, **kwargs) -> None:
        """
        Update metadata fields.
        
        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom[key] = value
        
        self.updated_at = datetime.now()


@dataclass
class Document:
    """
    Document class for storing content and embeddings.
    
    A document is the basic unit of storage in the memory store. It contains
    the content, metadata, and embeddings for a piece of text.
    """
    
    id: str
    content: str
    metadata: DocumentMetadata
    embedding: Optional[np.ndarray] = None
    chunks: List["Document"] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the document after initialization."""
        if isinstance(self.metadata, dict):
            self.metadata = DocumentMetadata(**self.metadata)
        
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)
    
    def update_embedding(self, embedding: Union[List[float], np.ndarray]) -> None:
        """
        Update the document's embedding.
        
        Args:
            embedding: New embedding vector
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        self.embedding = embedding
    
    def chunk(self, chunk_size: int, chunk_overlap: int = 0) -> List["Document"]:
        """
        Split the document into chunks.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List[Document]: List of document chunks
        """
        if len(self.content) <= chunk_size:
            return [self]
        
        chunks = []
        start = 0
        
        while start < len(self.content):
            end = min(start + chunk_size, len(self.content))
            
            # Try to find a natural break point (newline or period)
            if end < len(self.content):
                for break_char in ["\n", ".", " "]:
                    natural_break = self.content.rfind(break_char, start, end)
                    if natural_break != -1 and natural_break > start:
                        end = natural_break + 1
                        break
            
            chunk_content = self.content[start:end]
            chunk_id = f"{self.id}_chunk_{len(chunks)}"
            
            # Create metadata for the chunk
            chunk_metadata = DocumentMetadata(
                source=self.metadata.source,
                content_type=self.metadata.content_type,
                language=self.metadata.language,
                author=self.metadata.author,
                tags=self.metadata.tags.copy(),
                custom={
                    **self.metadata.custom,
                    "parent_id": self.id,
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end,
                },
            )
            
            chunk = Document(
                id=chunk_id,
                content=chunk_content,
                metadata=chunk_metadata,
                embedding=None,
            )
            
            chunks.append(chunk)
            start = end - chunk_overlap
        
        self.chunks = chunks
        return chunks
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the document
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.model_dump(),
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Create a document from a dictionary.
        
        Args:
            data: Dictionary representation of the document
            
        Returns:
            Document: Document instance
        """
        chunks_data = data.pop("chunks", [])
        doc = cls(**data)
        
        doc.chunks = [cls.from_dict(chunk_data) for chunk_data in chunks_data]
        return doc
