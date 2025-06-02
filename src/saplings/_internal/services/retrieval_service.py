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

from saplings.core.resilience import run_in_executor
from saplings.retrieval._internal import (
    CascadeRetriever,
    EmbeddingRetriever,
    EntropyCalculator,
    GraphExpander,
    RetrievalConfig,
    TFIDFRetriever,
)
from saplings.retrieval._internal.cached_embedding_retriever import CachedEmbeddingRetriever

if TYPE_CHECKING:
    from saplings.memory._internal import Document, MemoryStore

# Optional dependency – the service works even when monitoring is disabled.
try:
    from saplings.monitoring._internal.trace import TraceManager  # noqa
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
        self,
        query: str,
        *,
        limit: int | None = None,
        timeout: float | None = 60.0,  # Increased default timeout
        fast_mode: bool = False,
    ) -> list[Document]:
        """
        Retrieve up to *limit* documents relevant to *query* using the
        cascade retriever.

        Args:
        ----
            query: Search query
            limit: Maximum number of documents to return
            timeout: Optional timeout in seconds (default: 60.0)
            fast_mode: Whether to use fast retrieval mode (bypasses some steps for better performance)

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
                attributes={
                    "component": "retriever",
                    "query": query,
                    "limit": limit,
                    "fast_mode": fast_mode,
                },
            )

        try:
            # Use fast retrieval mode if requested
            if fast_mode:
                return await self._fast_retrieve(query, limit=limit, timeout=timeout)

            # Ensure TF-IDF index is built before retrieval to avoid building it during retrieval
            if not self.tfidf_retriever.is_built:
                logger.info("Building TF-IDF index before retrieval")

                def _build_index():
                    self.tfidf_retriever.build_index()

                try:
                    # Build index with a separate timeout
                    await run_in_executor(_build_index, timeout=timeout)
                except Exception as e:
                    logger.warning(f"Error building TF-IDF index: {e}")
                    # Continue anyway, the cascade retriever will handle it

            # Define function to run in executor
            def _retrieve_documents():
                try:
                    # Set a reasonable document limit if none is provided
                    document_limit = limit or min(self._cascade.config.entropy.max_documents, 50)

                    # Configure cascade retriever for better performance
                    original_max_iterations = self._cascade.config.entropy.max_iterations
                    original_max_documents = self._cascade.config.entropy.max_documents

                    # Temporarily adjust configuration for better performance
                    self._cascade.config.entropy.max_iterations = min(original_max_iterations, 2)
                    self._cascade.config.entropy.max_documents = min(
                        original_max_documents, document_limit
                    )

                    # Perform retrieval
                    result = self._cascade.retrieve(
                        query=query,
                        max_documents=document_limit,  # Use the parameter name expected by CascadeRetriever
                    )

                    # Restore original configuration
                    self._cascade.config.entropy.max_iterations = original_max_iterations
                    self._cascade.config.entropy.max_documents = original_max_documents

                    return list(result.documents)
                except Exception as e:
                    logger.error(f"Error in _retrieve_documents: {e}")
                    # Return empty list on error
                    return []

            # Run the blocking call in a thread pool with timeout
            documents = await run_in_executor(_retrieve_documents, timeout=timeout)

            logger.debug(
                "RetrievalService returned %d documents for query=%r",
                len(documents),
                query,
            )
            return documents
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            # Return empty list on error rather than propagating the exception
            return []
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    async def _fast_retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        timeout: float | None = 60.0,  # Increased default timeout
    ) -> list[Document]:
        """
        Fast retrieval mode that bypasses the full cascade retrieval process.

        This method uses only TF-IDF retrieval for better performance at the cost of
        potentially lower quality results.

        Args:
        ----
            query: Search query
            limit: Maximum number of documents to return
            timeout: Optional timeout in seconds (default: 60.0)

        Returns:
        -------
            List[Document]: Retrieved documents

        """
        span = None
        if self._trace_manager:
            trace = self._trace_manager.create_trace()
            span = self._trace_manager.start_span(
                name="RetrievalService._fast_retrieve",
                trace_id=trace.trace_id,
                attributes={"component": "retriever", "query": query, "limit": limit},
            )

        try:
            # Ensure TF-IDF index is built before retrieval
            if not self.tfidf_retriever.is_built:
                logger.info("Building TF-IDF index before fast retrieval")

                def _build_index():
                    self.tfidf_retriever.build_index()

                try:
                    # Build index with a separate timeout
                    await run_in_executor(_build_index, timeout=timeout)
                except Exception as e:
                    logger.warning(f"Error building TF-IDF index in fast retrieval: {e}")
                    # Continue anyway, we'll check again in _fast_retrieve_documents

            # Define function to run in executor
            def _fast_retrieve_documents():
                try:
                    # Ensure TF-IDF index is built
                    if not self.tfidf_retriever.is_built:
                        logger.info("Building TF-IDF index in _fast_retrieve_documents")
                        self.tfidf_retriever.build_index()

                    # Use TF-IDF retriever directly with a reasonable limit
                    k = limit or 20  # Default to 20 documents if limit is not specified
                    results = self.tfidf_retriever.retrieve(query=query, k=k)

                    # Extract documents
                    documents = [doc for doc, _ in results]

                    return documents
                except Exception as e:
                    logger.error(f"Error in _fast_retrieve_documents: {e}")
                    # Return empty list on error
                    return []

            # Run the blocking call in a thread pool with timeout
            documents = await run_in_executor(_fast_retrieve_documents, timeout=timeout)

            logger.debug(
                "Fast retrieval returned %d documents for query=%r",
                len(documents),
                query,
            )
            return documents
        except Exception as e:
            logger.error(f"Error in _fast_retrieve: {e}")
            # Return empty list on error rather than propagating the exception
            return []
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)
