from __future__ import annotations

"""
Retrieval Service Builder API module for Saplings.

This module provides the retrieval service builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class RetrievalServiceBuilder:
    """
    Builder for creating RetrievalService instances with a fluent interface.

    This builder provides a convenient way to configure and create RetrievalService
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the retrieval service builder."""
        self._memory_manager = None
        self._trace_manager = None
        self._config = {}

    def with_memory_manager(self, memory_manager: Any) -> "RetrievalServiceBuilder":
        """
        Set the memory manager for the retrieval service.

        Args:
        ----
            memory_manager: Memory manager for storing and retrieving documents

        Returns:
        -------
            Self for method chaining

        """
        self._memory_manager = memory_manager
        return self

    def with_trace_manager(self, trace_manager: Any) -> "RetrievalServiceBuilder":
        """
        Set the trace manager for the retrieval service.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_config(self, **kwargs) -> "RetrievalServiceBuilder":
        """
        Set additional configuration options for the retrieval service.

        Args:
        ----
            **kwargs: Additional configuration options

        Returns:
        -------
            Self for method chaining

        """
        self._config.update(kwargs)
        return self

    def build(self) -> Any:
        """
        Build the retrieval service instance.

        Returns
        -------
            RetrievalService instance

        """

        # Use a factory function to avoid circular imports
        def create_service():
            # Import the RetrievalConfig
            from saplings.api.core import RetrievalConfig
            from saplings.api.services.retrieval import RetrievalService

            # Create a config object
            config = RetrievalConfig(**self._config)

            # Create the service
            return RetrievalService(
                memory_manager=self._memory_manager,
                config=config,
                trace_manager=self._trace_manager,
            )

        return create_service()


__all__ = [
    "RetrievalServiceBuilder",
]
