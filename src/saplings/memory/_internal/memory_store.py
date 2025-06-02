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

from saplings.memory._internal.config import MemoryConfig, PrivacyLevel
from saplings.memory._internal.document import Document, DocumentMetadata
from saplings.memory._internal.graph.dependency_graph import DependencyGraph, DocumentNode
from saplings.memory._internal.indexing.indexer_registry import get_indexer

# Use internal implementation for vector_store
from saplings.memory._internal.vector_store.get_vector_store import get_vector_store

logger = logging.getLogger(__name__)

# Import sentence-transformers for embedding generation
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. Please install it with: pip install sentence-transformers"
    )


class MemoryStore:
    """
    Memory store for Saplings.

    The memory store combines vector storage and graph-based memory to provide
    efficient and context-aware retrieval of documents.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        indexer_registry=None,
        plugin_registry=None,
        vector_store=None,
        graph=None,
        indexer=None,
        embedding_model=None,
    ) -> None:
        """
        Initialize the memory store.

        Args:
        ----
            config: Memory configuration
            indexer_registry: Optional indexer registry to use
            plugin_registry: Optional plugin registry to use
            vector_store: Optional vector store to use
            graph: Optional dependency graph to use
            indexer: Optional indexer to use
            embedding_model: Optional embedding model to use

        """
        self.config = config or MemoryConfig.default()

        # Use provided components or create them
        self.vector_store = vector_store or get_vector_store(
            config=self.config, registry=plugin_registry
        )
        self.graph = graph or DependencyGraph(config=self.config)
        self.indexer = indexer or get_indexer(config=self.config, registry=indexer_registry)
        self.secure_mode = self.config.secure_store.privacy_level != PrivacyLevel.NONE

        # Initialize embedding model if not provided
        self.embedding_model = embedding_model
        if self.embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Initialized default embedding model: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
                self.embedding_model = None

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

        # Generate embedding if not provided and embedding model is available
        if document.embedding is None and self.embedding_model is not None:
            try:
                # Generate embedding
                generated_embedding = self.embedding_model.encode(content, show_progress_bar=False)

                # Convert to numpy array if needed
                if not isinstance(generated_embedding, np.ndarray):
                    generated_embedding = np.array(generated_embedding, dtype=np.float32)

                # Update document with generated embedding
                document.update_embedding(generated_embedding)
                logger.debug(f"Generated embedding for document {document.id}")
            except Exception as e:
                logger.warning(f"Failed to generate embedding for document {document.id}: {e}")

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
        # Generate embeddings for documents without embeddings
        if self.embedding_model is not None:
            docs_without_embeddings = [doc for doc in documents if doc.embedding is None]
            if docs_without_embeddings:
                try:
                    # Generate embeddings in batch
                    contents = [doc.content for doc in docs_without_embeddings]
                    batch_embeddings = self.embedding_model.encode(
                        contents, show_progress_bar=False
                    )

                    # Update documents with embeddings
                    for i, doc in enumerate(docs_without_embeddings):
                        embedding = batch_embeddings[i]
                        if not isinstance(embedding, np.ndarray):
                            embedding = np.array(embedding, dtype=np.float32)
                        doc.update_embedding(embedding)
                        logger.debug(f"Generated embedding for document {doc.id}")
                except Exception as e:
                    logger.warning(f"Failed to generate batch embeddings: {e}")

        # Apply security measures if enabled
        if self.secure_mode:
            documents = [self._secure_document(doc) for doc in documents]

        # Add documents to vector store
        docs_with_embeddings = [doc for doc in documents if doc.embedding is not None]
        if docs_with_embeddings:
            self.vector_store.add_documents(docs_with_embeddings)

        # Log warning for documents without embeddings
        docs_without_embeddings = [doc for doc in documents if doc.embedding is None]
        if docs_without_embeddings:
            logger.warning(
                f"{len(docs_without_embeddings)} documents have no embeddings and will not be added to the vector store."
            )

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
            # Convert entity to the correct type if needed
            from saplings.memory._internal.indexing.entity import Entity

            if not isinstance(entity, Entity):
                # Create a new entity with the same attributes
                entity_data = (
                    entity.to_dict()
                    if hasattr(entity, "to_dict")
                    else {"name": str(entity), "entity_type": "unknown"}
                )
                # Ensure metadata is a dict
                metadata = entity_data.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}

                entity = Entity(
                    name=entity_data.get("name", ""),
                    entity_type=entity_data.get("entity_type", "unknown"),
                    metadata=metadata,
                )
            self.graph.add_entity_node(entity)

        # Add relationships to graph
        for rel in indexing_result.relationships:
            try:
                # Convert relationship to the correct type if needed
                from saplings.memory._internal.graph.relationship import Relationship

                if not isinstance(rel, Relationship):
                    # Create a new relationship with the same attributes
                    rel_data = (
                        rel.to_dict()
                        if hasattr(rel, "to_dict")
                        else {"source_id": "", "target_id": "", "relationship_type": "unknown"}
                    )
                    # Ensure weight is a float
                    weight = rel_data.get("weight", 1.0)
                    if not isinstance(weight, (int, float)):
                        weight = 1.0

                    # Ensure metadata is a dict
                    metadata = rel_data.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}

                    relationship = Relationship(
                        source_id=rel_data.get("source_id", ""),
                        target_id=rel_data.get("target_id", ""),
                        relationship_type=rel_data.get("relationship_type", "unknown"),
                        weight=float(weight),
                        metadata=metadata,
                    )
                    self.graph.add_relationship(relationship)
                else:
                    self.graph.add_relationship(rel)
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

    def index_repository(
        self, repo_path: str, file_extensions: list[str] | None = None
    ) -> list[Document]:
        """
        Index a code repository by traversing its directory structure and adding code files.

        Args:
        ----
            repo_path: Path to the repository
            file_extensions: List of file extensions to include (default: common code file extensions)

        Returns:
        -------
            List[Document]: List of added documents

        Raises:
        ------
            FileNotFoundError: If the repository path does not exist

        """
        import os

        if not os.path.exists(repo_path):
            msg = f"Repository path does not exist: {repo_path}"
            raise FileNotFoundError(msg)

        if file_extensions is None:
            # Default to common code file extensions
            file_extensions = [
                ".py",
                ".js",
                ".ts",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".hpp",
                ".cs",
                ".go",
                ".rb",
                ".php",
                ".swift",
                ".kt",
                ".rs",
                ".scala",
            ]

        logger.info(f"Indexing repository at {repo_path} with extensions: {file_extensions}")

        documents = []

        # Walk through the repository
        for root, _, files in os.walk(repo_path):
            # Skip hidden directories and common directories to ignore
            if any(part.startswith(".") for part in root.split(os.sep)):
                continue

            # Skip common directories to ignore
            if any(
                ignore_dir in root.split(os.sep)
                for ignore_dir in ["node_modules", "venv", "env", "__pycache__", "dist", "build"]
            ):
                continue

            for file in files:
                # Check if the file has one of the target extensions
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)

                    try:
                        # Read the file content
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        # Create relative path for better organization
                        rel_path = os.path.relpath(file_path, repo_path)

                        # Add the document to memory
                        document = self.add_document(
                            content=content,
                            metadata={
                                "source": rel_path,
                                "file_path": file_path,
                                "type": "code",
                                "language": os.path.splitext(file)[1][
                                    1:
                                ],  # Extract language from extension
                            },
                        )

                        documents.append(document)
                        logger.debug(f"Indexed file: {rel_path}")

                    except Exception as e:
                        logger.warning(f"Error indexing file {file_path}: {e}")

        logger.info(f"Indexed {len(documents)} files from repository {repo_path}")
        return documents
