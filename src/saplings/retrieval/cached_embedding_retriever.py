from __future__ import annotations

"""
Cached embedding retriever module for Saplings.

This module provides a cached version of the embedding-based retriever
using the unified caching system.
"""


import logging
from typing import TYPE_CHECKING

import numpy as np

from saplings.core.caching import cached_embedding, cached_retrieval
from saplings.core.caching.interface import CacheStrategy
from saplings.core.caching.vector import embed_with_cache
from saplings.retrieval.embedding_retriever import EmbeddingRetriever

if TYPE_CHECKING:
    from saplings.memory.document import Document
    from saplings.retrieval.config import EmbeddingConfig, RetrievalConfig

logger = logging.getLogger(__name__)


class CachedEmbeddingRetriever(EmbeddingRetriever):
    """
    Cached embedding retriever for semantic search.

    This class extends the standard EmbeddingRetriever with caching capabilities
    using the unified caching system.
    """

    def __init__(
        self,
        memory_store,
        config: RetrievalConfig | EmbeddingConfig | None = None,
        cache_namespace: str = "vector",
        cache_ttl: int | None = 3600,
        cache_provider: str = "memory",
        cache_strategy: CacheStrategy = CacheStrategy.LRU,
    ) -> None:
        """
        Initialize the cached embedding retriever.

        Args:
        ----
            memory_store: Memory store containing the documents
            config: Retrieval or embedding configuration
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds
            cache_provider: Cache provider to use
            cache_strategy: Cache eviction strategy

        """
        super().__init__(memory_store, config)
        self.cache_namespace = cache_namespace
        self.cache_ttl = cache_ttl
        self.cache_provider = cache_provider
        self.cache_strategy = cache_strategy
        self.collection_name = "embedding_retriever"

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string with caching.

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

        # Define a safe encode function
        def safe_encode(q):
            if self.model is None:
                msg = "Embedding model not initialized"
                raise ValueError(msg)
            return self.model.encode(q, show_progress_bar=False)

        # Use the caching helper function
        embedding = embed_with_cache(
            embed_func=safe_encode,
            collection=self.collection_name,
            text=query,
            namespace=self.cache_namespace,
            ttl=self.cache_ttl,
            provider=self.cache_provider,
            strategy=self.cache_strategy,
        )

        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @cached_embedding(collection_attr="collection_name")
    def embed_single_document(self, text: str) -> np.ndarray:
        """
        Embed a single document with caching.

        Args:
        ----
            text: Document text to embed

        Returns:
        -------
            np.ndarray: Document embedding

        """
        if self.model is None:
            msg = "Embedding model not initialized"
            raise ValueError(msg)

        # Encode the document
        embedding = self.model.encode(text, show_progress_bar=False)

        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        # Threshold for norm
        NORM_THRESHOLD = 0
        if norm > NORM_THRESHOLD:
            embedding = embedding / norm

        return embedding

    def embed_documents(self, documents: list[Document]) -> dict[str, np.ndarray]:
        """
        Embed a list of documents with caching.

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
            # Process documents one by one to leverage caching
            for doc in docs_to_embed:
                embedding = self.embed_single_document(doc.content)
                embeddings[doc.id] = embedding

        return embeddings

    @cached_retrieval(collection_attr="collection_name")
    def retrieve(
        self,
        query: str,
        documents: list[Document],
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve documents similar to the query using embeddings with caching.

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
