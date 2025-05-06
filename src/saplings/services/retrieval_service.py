from __future__ import annotations

"""
saplings.services.retrieval_service.
===================================

Facade that groups all retrieval-related components into a single
injectable service.  Down-stream code should depend on this class rather
than instantiating individual retrievers.

Components
----------
* TF-IDF retriever
* Embedding retriever (with optional caching)
* Graph expander
* Entropy calculator
* Cascade retriever (high-level interface)
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.core.resilience import DEFAULT_TIMEOUT, run_in_executor
from saplings.retrieval import (
    CascadeRetriever,
    EmbeddingRetriever,
    EntropyCalculator,
    GraphExpander,
    RetrievalConfig,
    TFIDFRetriever,
)
from saplings.retrieval.cached_embedding_retriever import CachedEmbeddingRetriever

if TYPE_CHECKING:
    from saplings.memory import Document, MemoryStore

# Optional dependency – the service works even when monitoring is disabled.
try:
    from saplings.monitoring.trace import TraceManager  # noqa
except ModuleNotFoundError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


class RetrievalService:
    """High-level API over Saplings’ retrieval pipeline."""

    def __init__(
        self,
        memory_store: "MemoryStore",
        config: RetrievalConfig,
        trace_manager: Any = None,
    ) -> None:
        self._trace_manager = trace_manager

        # Low-level components
        self.tfidf_retriever = TFIDFRetriever(memory_store=memory_store, config=config)

        # Choose between cached or regular embedding retriever based on config
        if config.cache.enabled:
            logger.info(
                "Using cached embedding retriever with namespace: %s", config.cache.namespace
            )
            self.embedding_retriever = CachedEmbeddingRetriever(
                memory_store=memory_store,
                config=config,
                cache_namespace=config.cache.namespace,
                cache_ttl=config.cache.embedding_ttl,
                cache_provider=config.cache.provider,
                cache_strategy=config.cache.strategy,
            )
        else:
            logger.info("Using standard embedding retriever (caching disabled)")
            self.embedding_retriever = EmbeddingRetriever(memory_store=memory_store, config=config)

        self.graph_expander = GraphExpander(memory_store=memory_store, config=config)
        self.entropy_calculator = EntropyCalculator(config=config)

        # Cascade retriever orchestrates the above
        self._cascade = CascadeRetriever(memory_store=memory_store, config=config)

        logger.info(
            "RetrievalService initialised (entropy_threshold=%s, max_documents=%s)",
            config.entropy.threshold,
            config.entropy.max_documents,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    async def retrieve(
        self, query: str, *, limit: int | None = None, timeout: float | None = DEFAULT_TIMEOUT
    ) -> list[Document]:
        """
        Retrieve up to *limit* documents relevant to *query* using the
        cascade retriever.

        Args:
        ----
            query: Search query
            limit: Maximum number of documents to return
            timeout: Optional timeout in seconds

        Returns:
        -------
            List[Document]: Retrieved documents

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        span = None
        if self._trace_manager:
            trace = self._trace_manager.create_trace()
            span = self._trace_manager.start_span(
                name="RetrievalService.retrieve",
                trace_id=trace.trace_id,
                attributes={"component": "retriever", "query": query, "limit": limit},
            )

        try:
            # Define function to run in executor
            def _retrieve_documents():
                result = self._cascade.retrieve(
                    query=query,
                    max_documents=limit or self._cascade.config.entropy.max_documents,
                )
                return list(result.documents)

            # Run the blocking call in a thread pool with timeout
            documents = await run_in_executor(_retrieve_documents, timeout=timeout)

            logger.debug(
                "RetrievalService returned %d documents for query=%r",
                len(documents),
                query,
            )
            return documents
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)
