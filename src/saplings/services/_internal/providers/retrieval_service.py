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
from saplings.services._internal.base.lazy_service import LazyService


# Use lazy imports to avoid circular dependencies
def _get_retrieval_components():
    """Get retrieval components using lazy imports to avoid circular dependencies."""
    # Import from the public API
    from saplings.api.retrieval import (
        CachedEmbeddingRetriever,
        CascadeRetriever,
        EmbeddingRetriever,
        EntropyCalculator,
        GraphExpander,
        RetrievalConfig,
        TFIDFRetriever,
    )

    return (
        CachedEmbeddingRetriever,
        CascadeRetriever,
        EmbeddingRetriever,
        EntropyCalculator,
        GraphExpander,
        RetrievalConfig,
        TFIDFRetriever,
    )


if TYPE_CHECKING:
    from saplings.api.memory import MemoryStore
    from saplings.api.memory.document import Document
    from saplings.api.retrieval import RetrievalConfig

# Optional dependency – the service works even when monitoring is disabled.
try:
    from saplings.api.monitoring import TraceManager  # noqa
except ModuleNotFoundError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


class RetrievalService(LazyService):
    """
    High-level API over Saplings' retrieval pipeline with lazy initialization.

    This service provides functionality for retrieving documents from memory,
    including semantic search and filtering. It uses lazy initialization to
    defer the creation of expensive components until they are needed.
    """

    def __init__(
        self,
        memory_store: "MemoryStore",
        config: "RetrievalConfig",
        trace_manager: Any = None,
    ) -> None:
        """
        Initialize the retrieval service.

        Args:
        ----
            memory_store: Memory store for document retrieval
            config: Configuration for retrieval
            trace_manager: Optional trace manager for monitoring

        """
        super().__init__()
        self._memory_store = memory_store
        self._config = config
        self._trace_manager = trace_manager

        # Components will be initialized lazily
        self.tfidf_retriever = None
        self.embedding_retriever = None
        self.graph_expander = None
        self.entropy_calculator = None
        self._cascade = None

        logger.info("RetrievalService created (lazy initialization)")

    async def _initialize_impl(self) -> None:
        """
        Initialize the retrieval service components.

        This method is called by the LazyService base class during initialization.
        """
        logger.info("Initializing RetrievalService components")

        # Get retrieval components using lazy imports to avoid circular dependencies
        (
            CachedEmbeddingRetriever,
            CascadeRetriever,
            EmbeddingRetriever,
            EntropyCalculator,
            GraphExpander,
            _,  # RetrievalConfig
            TFIDFRetriever,
        ) = _get_retrieval_components()

        # Low-level components
        self.tfidf_retriever = TFIDFRetriever(memory_store=self._memory_store, config=self._config)

        # Choose between cached or regular embedding retriever based on config
        if self._config.cache.enabled:
            logger.info(
                "Using cached embedding retriever with namespace: %s", self._config.cache.namespace
            )
            self.embedding_retriever = CachedEmbeddingRetriever(
                memory_store=self._memory_store,
                config=self._config,
                cache_namespace=self._config.cache.namespace,
                cache_ttl=self._config.cache.embedding_ttl,
                cache_provider=self._config.cache.provider,
                cache_strategy=self._config.cache.strategy,
            )
        else:
            logger.info("Using standard embedding retriever (caching disabled)")
            self.embedding_retriever = EmbeddingRetriever(
                memory_store=self._memory_store, config=self._config
            )

        self.graph_expander = GraphExpander(memory_store=self._memory_store, config=self._config)
        self.entropy_calculator = EntropyCalculator(config=self._config)

        # Cascade retriever orchestrates the above
        self._cascade = CascadeRetriever(memory_store=self._memory_store, config=self._config)

        logger.info(
            "RetrievalService initialized (entropy_threshold=%s, max_documents=%s)",
            self._config.entropy.threshold,
            self._config.entropy.max_documents,
        )

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        timeout: float | None = 60.0,
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

        """
        # Ensure the service is initialized
        await self.initialize(timeout=timeout)

        # Create a span for tracing
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
        # Ensure the service is initialized
        await self.initialize(timeout=timeout)

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
