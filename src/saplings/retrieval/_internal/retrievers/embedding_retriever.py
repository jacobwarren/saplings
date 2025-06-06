from __future__ import annotations

"""
Embedding retriever module for Saplings.

This module provides the embedding-based retriever for semantic search.
"""


import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from saplings.retrieval._internal.config import EmbeddingConfig, RetrievalConfig

if TYPE_CHECKING:
    from saplings.api.memory import MemoryStore
    from saplings.api.memory.document import Document

logger = logging.getLogger(__name__)


class EmbeddingRetriever:
    """
    Embedding retriever for semantic search.

    This class uses embeddings to perform semantic search on documents.
    It refines the results from the TF-IDF retriever by re-ranking them
    based on semantic similarity.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        config: RetrievalConfig | EmbeddingConfig | None = None,
    ) -> None:
        """
        Initialize the embedding retriever.

        Args:
        ----
            memory_store: Memory store containing the documents
            config: Retrieval or embedding configuration

        """
        self.memory_store = memory_store

        # Extract embedding config from RetrievalConfig if needed
        if config is None:
            self.config = EmbeddingConfig(
                model_name="all-MiniLM-L6-v2",
                embedding_dimension=384,
                batch_size=32,
                similarity_top_k=20,
                similarity_cutoff=0.7,
                use_existing_embeddings=True,
            )
        elif isinstance(config, RetrievalConfig):
            self.config = config.embedding
        else:
            self.config = config

        # Initialize embedding model
        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.config.model_name)
            logger.info(f"Initialized embedding model: {self.config.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Please install it with: pip install sentence-transformers"
            )
            self.model = None

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.

        Args:
        ----
            query: Query string

        Returns:
        -------
            np.ndarray: Query embedding

        """
        if self.model is None:
            msg = "Embedding model not initialized"
            raise ValueError(msg)

        # Encode the query
        embedding = self.model.encode(query, show_progress_bar=False)

        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_documents(self, documents: list[Document]) -> dict[str, np.ndarray]:
        """
        Embed a list of documents.

        Args:
        ----
            documents: Documents to embed

        Returns:
        -------
            Dict[str, np.ndarray]: Dictionary mapping document IDs to embeddings

        """
        if self.model is None:
            msg = "Embedding model not initialized"
            raise ValueError(msg)

        # Filter documents that already have embeddings if configured
        docs_to_embed = []
        doc_indices = []
        embeddings = {}

        for i, doc in enumerate(documents):
            if self.config.use_existing_embeddings and doc.embedding is not None:
                embeddings[doc.id] = doc.embedding
            else:
                docs_to_embed.append(doc)
                doc_indices.append(i)

        if docs_to_embed:
            # Encode documents in batches
            batch_size = self.config.batch_size
            for i in range(0, len(docs_to_embed), batch_size):
                batch = docs_to_embed[i : i + batch_size]
                texts = [doc.content for doc in batch]
                batch_embeddings = self.model.encode(texts, show_progress_bar=False)

                # Process and store embeddings
                for j, doc in enumerate(batch):
                    # Get the embedding
                    embedding = batch_embeddings[j]

                    # Convert to numpy array if needed
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding, dtype=np.float32)

                    # Normalize the embedding
                    norm = np.linalg.norm(embedding)
                    # Threshold for norm
                    NORM_THRESHOLD = 0
                    if norm > NORM_THRESHOLD:
                        embedding = embedding / norm

                    # Store the normalized embedding
                    embeddings[doc.id] = embedding

        return embeddings

    def retrieve(
        self,
        query: str,
        documents: list[Document],
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents similar to the query using embeddings.

        Args:
        ----
            query: Query string
            documents: Documents to search (typically from TF-IDF retriever)
            k: Number of documents to retrieve (if None, uses config.similarity_top_k)

        Returns:
        -------
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples

        """
        if not documents:
            return []

        # Use default k if not provided
        if k is None:
            k = self.config.similarity_top_k

        # Embed query
        query_embedding = self.embed_query(query)

        # Embed documents
        document_embeddings = self.embed_documents(documents)

        # Calculate similarity scores
        results = []
        for doc in documents:
            if doc.id in document_embeddings:
                doc_embedding = document_embeddings[doc.id]
                similarity = float(np.dot(query_embedding, doc_embedding))

                # Apply similarity cutoff if configured
                if (
                    self.config.similarity_cutoff is None
                    or similarity >= self.config.similarity_cutoff
                ):
                    results.append((doc, similarity))

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return results[:k]

    def save(self, directory: str) -> None:
        """
        Save the embedding retriever configuration to disk.

        Args:
        ----
            directory: Directory to save to

        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(directory_path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f)

        logger.info(f"Saved embedding retriever configuration to {directory}")

    def load(self, directory: str) -> None:
        """
        Load the embedding retriever configuration from disk.

        Args:
        ----
            directory: Directory to load from

        """
        directory_path = Path(directory)

        # Load config
        config_path = directory_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                self.config = EmbeddingConfig(**config_data)

                # Re-initialize embedding model if model name changed
                self._initialize_embedding_model()

        logger.info(f"Loaded embedding retriever configuration from {directory}")
