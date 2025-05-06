from __future__ import annotations

"""
Memory store module for Saplings.

This module defines the MemoryStore class, which combines vector storage and
graph-based memory.
"""


import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from saplings.memory.config import MemoryConfig, PrivacyLevel
from saplings.memory.document import Document, DocumentMetadata
from saplings.memory.graph import DependencyGraph, DocumentNode
from saplings.memory.indexer import get_indexer
from saplings.memory.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Memory store for Saplings.

    The memory store combines vector storage and graph-based memory to provide
    efficient and context-aware retrieval of documents.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """
        Initialize the memory store.

        Args:
        ----
            config: Memory configuration

        """
        self.config = config or MemoryConfig.default()
        self.vector_store = get_vector_store(config=self.config)
        self.graph = DependencyGraph(config=self.config)
        self.indexer = get_indexer(config=self.config)
        self.secure_mode = self.config.secure_store.privacy_level != PrivacyLevel.NONE

    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | DocumentMetadata | None = None,
        document_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> Document:
        """
        Add a document to the memory store.

        Args:
        ----
            content: Document content
            metadata: Document metadata
            document_id: Optional document ID (generated if not provided)
            embedding: Optional embedding vector

        Returns:
        -------
            Document: Added document

        """
        # Create document ID if not provided
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Create metadata if not provided
        if metadata is None:
            metadata = DocumentMetadata(
                source=f"document:{document_id}",
                content_type="text",
                language="en",
                author="system",
            )
        elif isinstance(metadata, dict):
            metadata = DocumentMetadata(**metadata)

        # Create document
        document = Document(
            id=document_id,
            content=content,
            metadata=metadata,
            embedding=np.array(embedding) if embedding is not None else None,
        )

        # Apply security measures if enabled
        if self.secure_mode:
            document = self._secure_document(document)

        # Add document to vector store
        if document.embedding is None:
            logger.warning(
                f"Document {document.id} has no embedding. It will not be added to the vector store."
            )
        else:
            self.vector_store.add_document(document)

        # Add document to graph
        self.graph.add_document_node(document)

        # Index document if graph is enabled
        if self.config.graph.enable_graph:
            self._index_document(document)

        return document

    def add_documents(self, documents: list[Document], *, index: bool = True) -> list[Document]:
        """
        Add multiple documents to the memory store.

        Args:
        ----
            documents: Documents to add
            index: Whether to index the documents

        Returns:
        -------
            List[Document]: Added documents

        """
        # Apply security measures if enabled
        if self.secure_mode:
            documents = [self._secure_document(doc) for doc in documents]

        # Add documents to vector store
        docs_with_embeddings = [doc for doc in documents if doc.embedding is not None]
        if docs_with_embeddings:
            self.vector_store.add_documents(docs_with_embeddings)

        # Add documents to graph
        for document in documents:
            self.graph.add_document_node(document)

        # Index documents if graph is enabled
        if index and self.config.graph.enable_graph:
            for document in documents:
                self._index_document(document)

        return documents

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_dict: dict[str, Any] | None = None,
        *,
        include_graph_results: bool = True,
        max_graph_hops: int = 1,
    ) -> list[tuple[Document, float]]:
        """
        Search for documents similar to the query embedding.

        Args:
        ----
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filter_dict: Optional filter criteria
            include_graph_results: Whether to include graph-based results
            max_graph_hops: Maximum number of hops for graph expansion

        Returns:
        -------
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples

        """
        # Search in vector store
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
            filter_dict=filter_dict,
        )

        if not include_graph_results or not self.config.graph.enable_graph:
            return vector_results

        # Get document IDs from vector results
        doc_ids = [doc.id for doc, _ in vector_results]

        # Expand results using graph
        expanded_docs = set(doc_ids)
        graph_results = []

        # Get subgraph centered on the vector results
        if doc_ids:
            subgraph = self.graph.get_subgraph(
                node_ids=doc_ids,
                max_hops=max_graph_hops,
            )

            # Collect document nodes from the subgraph
            for node in subgraph.nodes.values():
                if isinstance(node, DocumentNode) and node.id not in expanded_docs:
                    document = node.document

                    # Calculate a score based on graph distance
                    # This is a simple heuristic and can be improved
                    score = 0.5  # Base score for graph results

                    graph_results.append((document, score))
                    expanded_docs.add(node.id)

        # Combine and sort results
        combined_results = vector_results + graph_results
        combined_results.sort(key=lambda x: x[1], reverse=True)

        return combined_results[:limit]

    def get_document(self, document_id: str) -> Document | None:
        """
        Get a document by ID.

        Args:
        ----
            document_id: Document ID

        Returns:
        -------
            Optional[Document]: Document if found, None otherwise

        """
        return self.vector_store.get(document_id)

    async def get_all_documents(self) -> list[Document]:
        """
        Get all documents in the memory store.

        Returns
        -------
            List[Document]: All documents in the memory store

        """
        return self.vector_store.list()

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the memory store.

        Args:
        ----
            document_id: Document ID

        Returns:
        -------
            bool: True if the document was deleted, False otherwise

        """
        # Delete from vector store
        deleted_from_vector = self.vector_store.delete(document_id)

        # Delete from graph
        node = self.graph.get_node(document_id)
        if node:
            # Remove all edges connected to this node
            neighbors = self.graph.get_neighbors(document_id, direction="both")
            for _neighbor in neighbors:
                # This is a simplified approach; a more efficient implementation
                # would directly remove edges from the graph
                pass

            # Remove the node
            self.graph.nodes.pop(document_id, None)

        return deleted_from_vector

    def update_document(
        self,
        document_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> Document | None:
        """
        Update a document in the memory store.

        Args:
        ----
            document_id: Document ID
            content: New content (if None, the content is not updated)
            metadata: New metadata (if None, the metadata is not updated)
            embedding: New embedding (if None, the embedding is not updated)

        Returns:
        -------
            Optional[Document]: Updated document if found, None otherwise

        """
        # Get the document
        document = self.vector_store.get(document_id)
        if not document:
            return None

        # Update content
        if content is not None:
            document.content = content

        # Update metadata
        if metadata is not None:
            if isinstance(metadata, dict):
                # Create a new DocumentMetadata with updated values
                if isinstance(document.metadata, DocumentMetadata):
                    # Create a dictionary from the existing metadata
                    metadata_dict = document.metadata.model_dump()
                    # Update with new values
                    metadata_dict.update(metadata)
                    # Create a new DocumentMetadata
                    document.metadata = DocumentMetadata(**metadata_dict)
                else:
                    # If metadata is not a DocumentMetadata, create a new one
                    document.metadata = DocumentMetadata(**metadata)
            else:
                document.metadata = metadata

        # Update embedding
        if embedding is not None:
            document.update_embedding(embedding)

        # Apply security measures if enabled
        if self.secure_mode:
            document = self._secure_document(document)

        # Update in vector store
        self.vector_store.update(document)

        # Re-index document if graph is enabled
        if self.config.graph.enable_graph:
            # First, remove existing relationships
            self._remove_document_relationships(document_id)

            # Then, re-index the document
            self._index_document(document)

        return document

    def clear(self) -> None:
        """Clear all data from the memory store."""
        self.vector_store.clear()
        self.graph = DependencyGraph(config=self.config)

    def save(self, directory: str) -> None:
        """
        Save the memory store to disk.

        Args:
        ----
            directory: Directory to save to

        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save vector store
        vector_store_dir = directory_path / "vector_store"
        self.vector_store.save(str(vector_store_dir))

        # Save graph
        graph_dir = directory_path / "graph"
        self.graph.save(str(graph_dir))

        # Save configuration
        with Path(directory_path / "config.json").open("w") as f:
            json.dump(self.config.model_dump(), f)

        logger.info("Saved memory store to %s", directory)

    def load(self, directory: str) -> None:
        """
        Load the memory store from disk.

        Args:
        ----
            directory: Directory to load from

        """
        directory_path = Path(directory)

        # Load configuration
        config_path = directory_path / "config.json"
        if config_path.exists():
            with Path(config_path).open() as f:
                config_data = json.load(f)
                self.config = MemoryConfig(**config_data)

        # Load vector store
        vector_store_dir = directory_path / "vector_store"
        if vector_store_dir.exists():
            self.vector_store = get_vector_store(config=self.config)
            self.vector_store.load(str(vector_store_dir))

        # Load graph
        graph_dir = directory_path / "graph"
        if graph_dir.exists():
            self.graph = DependencyGraph(config=self.config)
            self.graph.load(str(graph_dir))

        # Update secure mode
        self.secure_mode = self.config.secure_store.privacy_level != PrivacyLevel.NONE

        logger.info("Loaded memory store from %s", directory)

    def _index_document(self, document: Document) -> None:
        """
        Index a document to extract entities and relationships.

        Args:
        ----
            document: Document to index

        """
        if not self.config.graph.enable_graph:
            return

        # Extract entities and relationships
        indexing_result = self.indexer.index_document(document)

        # Add entities to graph
        for entity in indexing_result.entities:
            self.graph.add_entity_node(entity)

        # Add relationships to graph
        for relationship in indexing_result.relationships:
            try:
                self.graph.add_relationship(relationship)
            except ValueError as e:
                logger.warning("Failed to add relationship: %s", e)

    def _remove_document_relationships(self, document_id: str) -> None:
        """
        Remove all relationships involving a document.

        Args:
        ----
            document_id: Document ID

        """
        # This is a simplified implementation
        # A more efficient approach would directly remove edges from the graph
        node = self.graph.get_node(document_id)
        if not node:
            return

        # Get all neighbors
        self.graph.get_neighbors(document_id, direction="both")

        # Remove relationships
        # This is not implemented yet, as it requires modifying the graph structure

    def _secure_document(self, document: Document) -> Document:
        """
        Apply security measures to a document.

        Args:
        ----
            document: Document to secure

        Returns:
        -------
            Document: Secured document

        """
        privacy_level = self.config.secure_store.privacy_level

        if privacy_level == PrivacyLevel.NONE:
            return document

        # Create a copy of the document
        secured_doc = Document(
            id=document.id,
            content=document.content,
            metadata=document.metadata,
            embedding=document.embedding.copy() if document.embedding is not None else None,
        )

        # Apply hashing to document ID if needed
        if privacy_level in [PrivacyLevel.HASH_ONLY, PrivacyLevel.HASH_AND_DP]:
            salt = self.config.secure_store.hash_salt or ""
            secured_doc.id = self._hash_value(document.id, salt)

            # Hash metadata fields that might contain sensitive information
            if isinstance(secured_doc.metadata, DocumentMetadata):
                if hasattr(secured_doc.metadata, "source") and secured_doc.metadata.source:
                    # Create a new metadata object with hashed values
                    metadata_dict = secured_doc.metadata.model_dump()
                    metadata_dict["source"] = self._hash_value(secured_doc.metadata.source, salt)

                    if hasattr(secured_doc.metadata, "author") and secured_doc.metadata.author:
                        metadata_dict["author"] = self._hash_value(
                            secured_doc.metadata.author, salt
                        )

                    secured_doc.metadata = DocumentMetadata(**metadata_dict)

        # Apply differential privacy noise to embedding if needed
        if privacy_level == PrivacyLevel.HASH_AND_DP and secured_doc.embedding is not None:
            secured_doc.embedding = self._add_dp_noise(secured_doc.embedding)

        return secured_doc

    def _hash_value(self, value: str, salt: str) -> str:
        """
        Hash a value using SHA-256.

        Args:
        ----
            value: Value to hash
            salt: Salt to add to the hash

        Returns:
        -------
            str: Hashed value

        """
        return hashlib.sha256(f"{value}{salt}".encode()).hexdigest()

    def _add_dp_noise(self, embedding: np.ndarray) -> np.ndarray:
        """
        Add differential privacy noise to an embedding.

        Args:
        ----
            embedding: Embedding vector

        Returns:
        -------
            np.ndarray: Noisy embedding

        """
        epsilon = self.config.secure_store.dp_epsilon
        delta = self.config.secure_store.dp_delta
        sensitivity = self.config.secure_store.dp_sensitivity

        # Calculate noise scale based on epsilon, delta, and sensitivity
        # This is a simplified implementation of the Gaussian mechanism
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        # Generate Gaussian noise
        noise = np.random.default_rng().normal(0, noise_scale, embedding.shape)

        # Add noise to embedding
        noisy_embedding = embedding + noise

        # Normalize the embedding to maintain similarity properties
        norm = np.linalg.norm(noisy_embedding)

        # Threshold for norm
        norm_threshold = 0

        if norm > norm_threshold:
            noisy_embedding = noisy_embedding / norm

        return noisy_embedding
