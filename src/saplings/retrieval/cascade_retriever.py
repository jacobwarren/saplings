"""
Cascade retriever module for Saplings.

This module provides the cascade retriever that orchestrates the entire
retrieval pipeline with entropy-based termination.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from saplings.memory.document import Document
from saplings.memory.memory_store import MemoryStore
from saplings.retrieval.config import RetrievalConfig
from saplings.retrieval.embedding_retriever import EmbeddingRetriever
from saplings.retrieval.entropy_calculator import EntropyCalculator
from saplings.retrieval.graph_expander import GraphExpander
from saplings.retrieval.tfidf_retriever import TFIDFRetriever

logger = logging.getLogger(__name__)


class RetrievalResult:
    """
    Result of a retrieval operation.

    This class contains the documents retrieved, their scores, and metadata
    about the retrieval process.
    """

    def __init__(
        self,
        documents: List[Document],
        scores: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a retrieval result.

        Args:
            documents: Retrieved documents
            scores: Scores for each document
            metadata: Additional metadata about the retrieval
        """
        self.documents = documents
        self.scores = scores
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """Get the number of documents in the result."""
        return len(self.documents)

    def get_documents(self) -> List[Document]:
        """Get the retrieved documents."""
        return self.documents

    def get_scores(self) -> List[float]:
        """Get the scores for each document."""
        return self.scores

    def get_document_score_pairs(self) -> List[Tuple[Document, float]]:
        """Get document-score pairs."""
        return list(zip(self.documents, self.scores))

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the retrieval."""
        return self.metadata

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the retrieval result to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "scores": self.scores,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """
        Create a retrieval result from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            RetrievalResult: Retrieval result
        """
        from saplings.memory.document import Document

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
        config: Optional[RetrievalConfig] = None,
    ):
        """
        Initialize the cascade retriever.

        Args:
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

    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        max_documents: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve documents relevant to the query.

        Args:
            query: Query string
            filter_dict: Optional filter criteria
            max_documents: Maximum number of documents to retrieve

        Returns:
            RetrievalResult: Retrieval result
        """
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
        }

        # Set maximum documents
        if max_documents is None:
            max_documents = self.config.entropy.max_documents

        # Initialize result set
        all_documents: List[Document] = []
        all_scores: List[float] = []

        # Main retrieval loop
        iteration = 0
        while True:
            iteration += 1
            metadata["iterations"] = iteration

            # Step 1: TF-IDF retrieval
            tfidf_start = time.time()
            if iteration == 1:
                # Build TF-IDF index if not already built
                if not self.tfidf_retriever.is_built:
                    self.tfidf_retriever.build_index()

                # Initial TF-IDF retrieval
                tfidf_results = self.tfidf_retriever.retrieve(
                    query=query,
                    filter_dict=filter_dict,
                )
            else:
                # Expand TF-IDF retrieval in subsequent iterations
                tfidf_results = self.tfidf_retriever.retrieve(
                    query=query,
                    k=self.config.tfidf.initial_k * iteration,
                    filter_dict=filter_dict,
                )

            tfidf_docs = [doc for doc, _ in tfidf_results]
            tfidf_scores = [score for _, score in tfidf_results]
            metadata["tfidf_time"] += time.time() - tfidf_start

            # Step 2: Embedding-based retrieval
            embedding_start = time.time()
            embedding_results = self.embedding_retriever.retrieve(
                query=query,
                documents=tfidf_docs,
                k=self.config.embedding.similarity_top_k,
            )

            embedding_docs = [doc for doc, _ in embedding_results]
            embedding_scores = [score for _, score in embedding_results]
            metadata["embedding_time"] += time.time() - embedding_start

            # Step 3: Graph expansion
            graph_start = time.time()
            graph_results = self.graph_expander.expand(
                documents=embedding_docs,
                scores=embedding_scores,
            )

            graph_docs = [doc for doc, _ in graph_results]
            graph_scores = [score for _, score in graph_results]
            metadata["graph_time"] += time.time() - graph_start

            # Merge results with existing set
            current_doc_ids = {doc.id for doc in all_documents}
            for doc, score in graph_results:
                if doc.id not in current_doc_ids:
                    all_documents.append(doc)
                    all_scores.append(score)
                    current_doc_ids.add(doc.id)

            # Limit to max_documents
            if len(all_documents) > max_documents:
                # Sort by score and take top max_documents
                doc_score_pairs = sorted(
                    zip(all_documents, all_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                all_documents = [doc for doc, _ in doc_score_pairs[:max_documents]]
                all_scores = [score for _, score in doc_score_pairs[:max_documents]]

            # Step 4: Check termination condition
            entropy_start = time.time()
            should_terminate = self.entropy_calculator.should_terminate(
                documents=all_documents,
                iteration=iteration,
            )
            metadata["entropy_time"] += time.time() - entropy_start

            if should_terminate:
                break

        # Calculate final entropy
        final_entropy = self.entropy_calculator.calculate_entropy(all_documents)
        metadata["final_entropy"] = final_entropy
        metadata["end_time"] = time.time()
        metadata["total_time"] = metadata["end_time"] - metadata["start_time"]

        # Create result
        result = RetrievalResult(
            documents=all_documents,
            scores=all_scores,
            metadata=metadata,
        )

        return result

    def save(self, directory: str) -> None:
        """
        Save the cascade retriever to disk.

        Args:
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
            directory: Directory to load from
        """
        directory_path = Path(directory)

        # Load config
        config_path = directory_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
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
