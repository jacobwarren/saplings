"""
SecureMemoryStore plugin for Saplings.

This module provides a secure memory store implementation that uses
hash-key protection and differential privacy noise for privacy.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from saplings.core.plugin import MemoryStorePlugin, PluginType
from saplings.memory.config import MemoryConfig, PrivacyLevel
from saplings.memory.document import Document
from saplings.memory.vector_store import InMemoryVectorStore, VectorStore

logger = logging.getLogger(__name__)


class SecureMemoryStore(MemoryStorePlugin, VectorStore):
    """
    Secure memory store implementation.

    This memory store provides privacy protection through hash-key protection
    and differential privacy noise.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the secure memory store.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig.default()
        self._inner_store = InMemoryVectorStore(config)
        self.privacy_level = self.config.secure_store.privacy_level
        self.hash_salt = self.config.secure_store.hash_salt or "saplings-secure"
        self.dp_epsilon = self.config.secure_store.dp_epsilon

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "secure_memory_store"

    @property
    def version(self) -> str:
        """Version of the plugin."""
        return "0.1.0"

    @property
    def description(self) -> str:
        """Description of the plugin."""
        return "Secure memory store with hash-key protection and differential privacy noise"

    @property
    def plugin_type(self) -> PluginType:
        """Type of the plugin."""
        from saplings.core.plugin import PluginType

        return PluginType.MEMORY_STORE

    def _hash_document_content(self, content: str) -> str:
        """
        Hash document content for privacy protection.

        Args:
            content: Document content

        Returns:
            str: Hashed content
        """
        if self.privacy_level in [PrivacyLevel.HASH, PrivacyLevel.HASH_AND_DP]:
            # Create a salted hash of the content
            hash_obj = hashlib.sha256((content + self.hash_salt).encode())
            return hash_obj.hexdigest()

        return content

    def _add_dp_noise(self, embedding: np.ndarray) -> np.ndarray:
        """
        Add differential privacy noise to an embedding.

        Args:
            embedding: Embedding vector

        Returns:
            np.ndarray: Embedding with noise
        """
        if self.privacy_level in [PrivacyLevel.DP_NOISE, PrivacyLevel.HASH_AND_DP]:
            # Add Laplace noise with scale 1/epsilon
            noise_scale = 1.0 / self.dp_epsilon
            noise = np.random.laplace(0, noise_scale, embedding.shape)
            return embedding + noise

        return embedding

    def _secure_document(self, document: Document) -> Document:
        """
        Apply security measures to a document.

        Args:
            document: Document to secure

        Returns:
            Document: Secured document
        """
        # Create a copy of the document
        secured_doc = Document(
            id=document.id,
            content=self._hash_document_content(document.content),
            metadata=document.metadata,
            embedding=document.embedding,
            chunks=document.chunks,
        )

        # Apply DP noise to embedding if present
        if secured_doc.embedding is not None:
            secured_doc.embedding = self._add_dp_noise(secured_doc.embedding)

        # Apply security to chunks
        secured_chunks = []
        for chunk in secured_doc.chunks:
            secured_chunk = Document(
                id=chunk.id,
                content=self._hash_document_content(chunk.content),
                metadata=chunk.metadata,
                embedding=chunk.embedding,
            )

            # Apply DP noise to chunk embedding if present
            if secured_chunk.embedding is not None:
                secured_chunk.embedding = self._add_dp_noise(secured_chunk.embedding)

            secured_chunks.append(secured_chunk)

        secured_doc.chunks = secured_chunks

        return secured_doc

    def add_document(self, document: Document) -> None:
        """
        Add a document to the vector store.

        Args:
            document: Document to add
        """
        # Apply security measures
        secured_doc = self._secure_document(document)

        # Add to inner store
        self._inner_store.add_document(secured_doc)

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to the vector store.

        Args:
            documents: Documents to add
        """
        # Apply security measures to all documents
        secured_docs = [self._secure_document(doc) for doc in documents]

        # Add to inner store
        self._inner_store.add_documents(secured_docs)

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_dict: Optional filter criteria

        Returns:
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples
        """
        # Add DP noise to query embedding for privacy
        secured_embedding = self._add_dp_noise(query_embedding)

        # Search inner store
        return self._inner_store.search(secured_embedding, limit, filter_dict)

    def get(self, document_id: str) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            document_id: ID of the document

        Returns:
            Optional[Document]: The document, or None if not found
        """
        return self._inner_store.get(document_id)

    def delete(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            document_id: ID of the document to delete

        Returns:
            bool: True if the document was deleted, False otherwise
        """
        return self._inner_store.delete(document_id)

    def clear(self) -> None:
        """Clear all data from the vector store."""
        self._inner_store.clear()

    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.

        Args:
            directory: Directory to save to
        """
        self._inner_store.save(directory)

    def load(self, directory: str) -> None:
        """
        Load the vector store from disk.

        Args:
            directory: Directory to load from
        """
        self._inner_store.load(directory)

    def count(self) -> int:
        """
        Get the number of documents in the vector store.

        Returns:
            int: Number of documents
        """
        return len(self._inner_store.documents)

    def list(self) -> List[str]:
        """
        List all document IDs in the vector store.

        Returns:
            List[str]: List of document IDs
        """
        return list(self._inner_store.documents.keys())

    def update(self, document: Document) -> None:
        """
        Update a document in the vector store.

        Args:
            document: Document to update
        """
        # Apply security measures
        secured_doc = self._secure_document(document)

        # Delete the old document
        self.delete(document.id)

        # Add the updated document
        self._inner_store.add_document(secured_doc)
