from __future__ import annotations

"""
Cascade retriever module for Saplings.

This module provides the cascade retriever that orchestrates the entire
retrieval pipeline with entropy-based termination.
"""


import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from saplings.retrieval._internal.config import RetrievalConfig
from saplings.retrieval._internal.expansion.entropy_calculator import EntropyCalculator
from saplings.retrieval._internal.expansion.graph_expander import GraphExpander
from saplings.retrieval._internal.retrievers.embedding_retriever import EmbeddingRetriever
from saplings.retrieval._internal.retrievers.tfidf_retriever import TFIDFRetriever

if TYPE_CHECKING:
    from saplings.api.memory import MemoryStore
    from saplings.api.memory.document import Document

logger = logging.getLogger(__name__)


class RetrievalResult:
    """
    Result of a retrieval operation.

    This class contains the documents retrieved, their scores, and metadata
    about the retrieval process.
    """

    def __init__(
        self,
        documents: list[Document],
        scores: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a retrieval result.

        Args:
        ----
            documents: Retrieved documents
            scores: Scores for each document
            metadata: Additional metadata about the retrieval

        """
        self.documents = documents
        self.scores = scores
        self.metadata = metadata or {}

    def __len__(self):
        """Get the number of documents in the result."""
        return len(self.documents)

    def get_documents(self):
        """Get the retrieved documents."""
        return self.documents

    def get_scores(self):
        """Get the scores for each document."""
        return self.scores

    def get_document_score_pairs(self):
        """Get document-score pairs."""
        return list(zip(self.documents, self.scores))

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about the retrieval."""
        return self.metadata

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the retrieval result to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation

        """
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "scores": self.scores,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        """
        Create a retrieval result from a dictionary.

        Args:
        ----
            data: Dictionary representation

        Returns:
        -------
            RetrievalResult: Retrieval result

        """
        from saplings.api.memory.document import Document

        documents = [Document.from_dict(doc_data) for doc_data in data["documents"]]
        scores = data["scores"]
        metadata = data.get("metadata", {})
        return cls(documents=documents, scores=scores, metadata=metadata)


class CascadeRetriever:
    """
    Cascade retriever for orchestrating the retrieval pipeline.

    This class implements the cascaded retrieval pipeline (TF-IDF → embeddings → graph)
    with entropy-based termination.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        """
        Initialize the cascade retriever.

        Args:
        ----
            memory_store: Memory store containing the documents
            config: Retrieval configuration

        """
        self.memory_store = memory_store
        self.config = config or RetrievalConfig.default()

        # Initialize components
        self.tfidf_retriever = TFIDFRetriever(memory_store, self.config)
        self.embedding_retriever = EmbeddingRetriever(memory_store, self.config)
        self.graph_expander = GraphExpander(memory_store, self.config)
        self.entropy_calculator = EntropyCalculator(self.config)

    # Cache for query results to avoid redundant computation
    # The cache is keyed by (query, filter_dict_str, limit)
    _query_cache = {}

    def _get_cache_key(self, query: str, filter_dict: dict[str, Any] | None, limit: int) -> tuple:
        """Generate a cache key for the query."""
        filter_str = json.dumps(filter_dict, sort_keys=True) if filter_dict else "None"
        return (query, filter_str, limit)

    def retrieve(
        self,
        query: str,
        filter_dict: dict[str, Any] | None = None,
        limit: int | None = None,
        use_cache: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve documents relevant to the query.

        Args:
        ----
            query: Query string
            filter_dict: Optional filter criteria
            limit: Maximum number of documents to retrieve
            use_cache: Whether to use the query cache (default: True)

        Returns:
        -------
            RetrievalResult: Retrieval result

        """
        # Set document limit
        document_limit = limit

        # Set maximum documents if not provided
        if document_limit is None:
            document_limit = self.config.entropy.max_documents

        # Check cache if enabled
        if use_cache and self.config.cache.enabled:
            cache_key = self._get_cache_key(
                query, filter_dict, document_limit or 0
            )  # Use 0 as default if None
            cached_result = self._query_cache.get(cache_key)
            if cached_result:
                logger.debug(f"Using cached result for query: {query[:50]}...")
                return cached_result

        # Reset entropy calculator
        self.entropy_calculator.reset()

        # Initialize metadata
        metadata = {
            "query": query,
            "filter": filter_dict,
            "start_time": time.time(),
            "iterations": 0,
            "tfidf_time": 0.0,
            "embedding_time": 0.0,
            "graph_time": 0.0,
            "entropy_time": 0.0,
            "document_count": 0,
            "is_large_collection": False,
        }

        # Initialize result set
        all_documents: list[Document] = []
        all_scores: list[float] = []
        current_doc_ids = set()  # Track document IDs for faster lookups

        # Ensure TF-IDF index is built before starting retrieval
        if not self.tfidf_retriever.is_built:
            logger.info("Building TF-IDF index before retrieval")
            tfidf_start = time.time()
            self.tfidf_retriever.build_index()
            metadata["tfidf_time"] += time.time() - tfidf_start

        # Check if we're dealing with a large collection
        # Use vector_store.list() directly since get_all_documents() is async
        collection_size = len(self.memory_store.vector_store.list())
        is_large_collection = collection_size > 1000
        metadata["is_large_collection"] = is_large_collection
        metadata["collection_size"] = collection_size

        # Adjust parameters for large collections
        max_iterations = self.config.entropy.max_iterations
        if is_large_collection:
            # For large collections, use more aggressive early stopping
            max_iterations = min(max_iterations, 2)
            logger.debug(
                f"Large collection detected ({collection_size} documents), limiting to {max_iterations} iterations"
            )

        # Main retrieval loop
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            metadata["iterations"] = iteration

            logger.debug(f"Retrieval iteration {iteration}/{max_iterations}")

            # Step 1: TF-IDF retrieval with optimized parameters for large collections
            tfidf_start = time.time()

            # Determine k value based on collection size and iteration
            k_value = self.config.tfidf.initial_k
            if is_large_collection:
                # For large collections, use a larger initial k but don't increase it as much in later iterations
                k_value = min(self.config.tfidf.initial_k * 2, 200)
            elif iteration > 1:
                # For smaller collections, increase k in subsequent iterations if we need more documents
                if len(all_documents) < document_limit // 2:
                    k_value = min(self.config.tfidf.initial_k * iteration, 200)  # Cap at 200

            # Perform TF-IDF retrieval
            tfidf_results = self.tfidf_retriever.retrieve(
                query=query,
                k=k_value,
                filter_dict=filter_dict,
            )

            tfidf_docs = [doc for doc, _ in tfidf_results]
            metadata["tfidf_time"] += time.time() - tfidf_start

            # If no documents were retrieved, return an empty result
            if not tfidf_docs and iteration == 1:
                logger.warning("No documents retrieved from TF-IDF retriever")
                metadata["end_time"] = time.time()
                metadata["total_time"] = metadata["end_time"] - metadata["start_time"]
                result = RetrievalResult(
                    documents=[],
                    scores=[],
                    metadata=metadata,
                )

                # Cache the result if caching is enabled
                if use_cache and self.config.cache.enabled:
                    cache_key = self._get_cache_key(
                        query, filter_dict, document_limit or 0
                    )  # Use 0 as default if None
                    self._query_cache[cache_key] = result

                return result

            # Step 2: Embedding-based retrieval with optimizations
            # Skip embedding retrieval if we already have enough documents
            if len(all_documents) >= document_limit and iteration > 1:
                logger.debug(
                    f"Skipping embedding retrieval (already have {len(all_documents)} documents)"
                )
                embedding_docs = []
                embedding_scores = []
            else:
                embedding_start = time.time()

                # For large collections, limit the number of documents sent to embedding retrieval
                embedding_input_docs = tfidf_docs
                if is_large_collection and len(tfidf_docs) > 50:
                    # Take only the top 50 TF-IDF results for embedding retrieval
                    embedding_input_docs = tfidf_docs[:50]
                    logger.debug(
                        "Large collection: limiting embedding input to top 50 TF-IDF results"
                    )

                # Determine k value for embedding retrieval
                embedding_k = min(self.config.embedding.similarity_top_k, 50)  # Cap at 50
                if is_large_collection:
                    # For large collections, be more selective
                    embedding_k = min(embedding_k, 30)

                embedding_results = self.embedding_retriever.retrieve(
                    query=query,
                    documents=embedding_input_docs,
                    k=embedding_k,
                )

                embedding_docs = [doc for doc, _ in embedding_results]
                embedding_scores = [score for _, score in embedding_results]
                metadata["embedding_time"] += time.time() - embedding_start

            # Step 3: Graph expansion with optimizations
            # Skip graph expansion if we already have enough documents or if embedding retrieval was skipped
            if not embedding_docs or (len(all_documents) >= document_limit and iteration > 1):
                logger.debug("Skipping graph expansion")
                graph_results = []
            else:
                graph_start = time.time()

                # For large collections, limit graph expansion
                graph_max_nodes = self.config.graph.max_nodes
                if is_large_collection:
                    # Reduce the number of nodes to expand for large collections
                    graph_max_nodes = min(graph_max_nodes, 30)

                # Store original value
                original_max_nodes = self.config.graph.max_nodes

                # Temporarily modify config
                self.config.graph.max_nodes = graph_max_nodes

                graph_results = self.graph_expander.expand(
                    documents=embedding_docs,
                    scores=embedding_scores,
                )

                # Restore original value
                self.config.graph.max_nodes = original_max_nodes

                metadata["graph_time"] += time.time() - graph_start

            # Merge results with existing set more efficiently
            # First add TF-IDF results if we don't have enough documents
            if len(all_documents) < document_limit:
                for doc, score in tfidf_results:
                    if doc.id not in current_doc_ids:
                        all_documents.append(doc)
                        all_scores.append(score)
                        current_doc_ids.add(doc.id)

            # Then add graph results
            for doc, score in graph_results:
                if doc.id not in current_doc_ids:
                    all_documents.append(doc)
                    all_scores.append(score)
                    current_doc_ids.add(doc.id)

            # Limit to document_limit more efficiently
            if len(all_documents) > document_limit:
                # Use a more efficient approach for large document sets
                if len(all_documents) > 1000:
                    # For very large sets, use a faster approach that avoids full sorting
                    # Find the score threshold for the top document_limit
                    threshold_idx = len(all_scores) - document_limit
                    if threshold_idx > 0:
                        import numpy as np

                        threshold = np.partition(all_scores, threshold_idx)[threshold_idx]

                        # Keep only documents with scores above the threshold
                        filtered_docs = []
                        filtered_scores = []
                        for doc, score in zip(all_documents, all_scores):
                            if score >= threshold:
                                filtered_docs.append(doc)
                                filtered_scores.append(score)

                        # Sort the filtered documents by score
                        sorted_pairs = sorted(
                            zip(filtered_docs, filtered_scores), key=lambda x: x[1], reverse=True
                        )
                        all_documents = [doc for doc, _ in sorted_pairs[:document_limit]]
                        all_scores = [score for _, score in sorted_pairs[:document_limit]]
                else:
                    # For smaller sets, use the original approach
                    doc_score_pairs = sorted(
                        zip(all_documents, all_scores),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    all_documents = [doc for doc, _ in doc_score_pairs[:document_limit]]
                    all_scores = [score for _, score in doc_score_pairs[:document_limit]]

                # Update the current_doc_ids set to match the filtered documents
                current_doc_ids = {doc.id for doc in all_documents}

            # Step 4: Check termination conditions with optimizations for large collections
            # Early termination if we have enough documents
            if len(all_documents) >= document_limit:
                if iteration > 1 or is_large_collection:
                    logger.debug(
                        f"Early termination: have {len(all_documents)} documents (max={document_limit})"
                    )
                    break

            # For large collections, we might want to terminate after the first iteration
            # if we have a reasonable number of documents
            if is_large_collection and iteration == 1 and len(all_documents) >= document_limit // 2:
                logger.debug(
                    f"Large collection early termination: have {len(all_documents)} documents"
                )
                break

            # Check entropy-based termination
            entropy_start = time.time()
            should_terminate = self.entropy_calculator.should_terminate(
                documents=all_documents,
                iteration=iteration,
            )
            metadata["entropy_time"] += time.time() - entropy_start

            if should_terminate:
                logger.debug(f"Entropy-based termination after iteration {iteration}")
                break

        # Calculate final entropy
        final_entropy = self.entropy_calculator.calculate_entropy(all_documents)
        metadata["final_entropy"] = final_entropy
        metadata["end_time"] = time.time()
        metadata["total_time"] = metadata["end_time"] - metadata["start_time"]
        metadata["document_count"] = len(all_documents)

        logger.info(
            f"Retrieval completed in {metadata['total_time']:.2f}s with {len(all_documents)} documents "
            f"after {iteration} iterations (max={max_iterations})"
        )

        # Create result
        result = RetrievalResult(
            documents=all_documents,
            scores=all_scores,
            metadata=metadata,
        )

        # Cache the result if caching is enabled
        if use_cache and self.config.cache.enabled:
            cache_key = self._get_cache_key(
                query, filter_dict, document_limit or 0
            )  # Use 0 as default if None
            self._query_cache[cache_key] = result

            # Limit cache size
            if len(self._query_cache) > self.config.cache.max_size:
                # Remove oldest entries (simple LRU implementation)
                excess = len(self._query_cache) - self.config.cache.max_size
                for _ in range(excess):
                    self._query_cache.pop(next(iter(self._query_cache)))

        return result

    def save(self, directory: str) -> None:
        """
        Save the cascade retriever to disk.

        Args:
        ----
            directory: Directory to save to

        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save components
        self.tfidf_retriever.save(str(directory_path / "tfidf"))
        self.embedding_retriever.save(str(directory_path / "embedding"))
        self.graph_expander.save(str(directory_path / "graph"))
        self.entropy_calculator.save(str(directory_path / "entropy"))

        # Save config
        with open(directory_path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f)

        logger.info(f"Saved cascade retriever to {directory}")

    def load(self, directory: str) -> None:
        """
        Load the cascade retriever from disk.

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
                self.config = RetrievalConfig(**config_data)

        # Load components
        tfidf_path = directory_path / "tfidf"
        if tfidf_path.exists():
            self.tfidf_retriever.load(str(tfidf_path))

        embedding_path = directory_path / "embedding"
        if embedding_path.exists():
            self.embedding_retriever.load(str(embedding_path))

        graph_path = directory_path / "graph"
        if graph_path.exists():
            self.graph_expander.load(str(graph_path))

        entropy_path = directory_path / "entropy"
        if entropy_path.exists():
            self.entropy_calculator.load(str(entropy_path))

        logger.info(f"Loaded cascade retriever from {directory}")
