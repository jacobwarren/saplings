from __future__ import annotations

"""
Retrieval Service Builder module for Saplings.

This module provides a builder class for creating RetrievalService instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
from typing import Any

# Import from public API
from saplings.api.retrieval import RetrievalConfig
from saplings.services._internal.base.lazy_service_builder import LazyServiceBuilder
from saplings.services._internal.providers.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class RetrievalServiceBuilder(LazyServiceBuilder[RetrievalService]):
    """
    Builder for RetrievalService.

    This class provides a fluent interface for building RetrievalService instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for RetrievalService
    builder = RetrievalServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_memory_store(memory_store) \
                    .with_entropy_threshold(0.1) \
                    .with_max_documents(10) \
                    .with_trace_manager(trace_manager) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the retrieval service builder."""
        super().__init__(RetrievalService)
        self._memory_store = None
        self._entropy_threshold = 0.1
        self._max_documents = 10
        self._max_iterations = 3
        self._min_documents = 5
        self._use_normalized_entropy = True
        self._window_size = 3
        self._trace_manager = None
        self.require_dependency("memory_store")

    def with_memory_store(self, memory_store: Any) -> RetrievalServiceBuilder:
        """
        Set the memory store.

        Args:
        ----
            memory_store: Memory store for document retrieval

        Returns:
        -------
            The builder instance for method chaining

        """
        self._memory_store = memory_store
        self.with_dependency("memory_store", memory_store)
        return self

    def with_entropy_threshold(self, threshold: float) -> RetrievalServiceBuilder:
        """
        Set the entropy threshold.

        Args:
        ----
            threshold: Entropy threshold for retrieval

        Returns:
        -------
            The builder instance for method chaining

        """
        self._entropy_threshold = threshold
        return self

    def with_max_documents(self, max_documents: int) -> RetrievalServiceBuilder:
        """
        Set the maximum number of documents to retrieve.

        Args:
        ----
            max_documents: Maximum number of documents

        Returns:
        -------
            The builder instance for method chaining

        """
        self._max_documents = max_documents
        return self

    def with_trace_manager(self, trace_manager: Any) -> RetrievalServiceBuilder:
        """
        Set the trace manager.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            The builder instance for method chaining

        """
        self._trace_manager = trace_manager
        self.with_dependency("trace_manager", trace_manager)
        return self

    def with_lazy_initialization(self, lazy: bool) -> "RetrievalServiceBuilder":
        """
        Set whether to use lazy initialization.

        Args:
        ----
            lazy: Whether to use lazy initialization

        Returns:
        -------
            The builder instance for method chaining

        """
        super().with_lazy_initialization(lazy)
        return self

    def build(self) -> RetrievalService:
        """
        Build the retrieval service instance with the configured dependencies.

        Returns
        -------
            The initialized retrieval service instance

        """
        # Create the retrieval config with entropy settings
        from saplings.api.retrieval import EntropyConfig

        entropy_config = EntropyConfig(
            threshold=self._entropy_threshold,
            max_documents=self._max_documents,
            max_iterations=self._max_iterations,
            min_documents=self._min_documents,
            use_normalized_entropy=self._use_normalized_entropy,
            window_size=self._window_size,
        )

        config = RetrievalConfig(entropy=entropy_config)
        self.with_dependency("config", config)

        return super().build()
