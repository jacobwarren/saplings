from __future__ import annotations

"""
Memory Manager Builder API module for Saplings.

This module provides the memory manager builder implementation.
"""

from typing import Any

from saplings.api.stability import stable


@stable
class MemoryManagerBuilder:
    """
    Builder for creating MemoryManager instances with a fluent interface.

    This builder provides a convenient way to configure and create MemoryManager
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the memory manager builder."""
        self._vector_store = None
        self._dependency_graph = None
        self._trace_manager = None
        self._config = {}

    def with_vector_store(self, vector_store: Any) -> "MemoryManagerBuilder":
        """
        Set the vector store for the memory manager.

        Args:
        ----
            vector_store: Vector store for storing and retrieving embeddings

        Returns:
        -------
            Self for method chaining

        """
        self._vector_store = vector_store
        return self

    def with_dependency_graph(self, dependency_graph: Any) -> "MemoryManagerBuilder":
        """
        Set the dependency graph for the memory manager.

        Args:
        ----
            dependency_graph: Dependency graph for tracking relationships

        Returns:
        -------
            Self for method chaining

        """
        self._dependency_graph = dependency_graph
        return self

    def with_trace_manager(self, trace_manager: Any) -> "MemoryManagerBuilder":
        """
        Set the trace manager for the memory manager.

        Args:
        ----
            trace_manager: Trace manager for monitoring

        Returns:
        -------
            Self for method chaining

        """
        self._trace_manager = trace_manager
        return self

    def with_config(self, **kwargs) -> "MemoryManagerBuilder":
        """
        Set additional configuration options for the memory manager.

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
        Build the memory manager instance.

        Returns
        -------
            MemoryManager instance

        """

        # Use a factory function to avoid circular imports
        def create_manager():
            # Import directly from the service module to avoid circular imports
            # Import the MemoryConfig
            from saplings.api.core import MemoryConfig
            from saplings.api.services.memory import MemoryManager

            # Create a config object
            config = MemoryConfig(**self._config)

            # Create the manager
            return MemoryManager(
                vector_store=self._vector_store,
                dependency_graph=self._dependency_graph,
                config=config,
                trace_manager=self._trace_manager,
            )

        return create_manager()


__all__ = [
    "MemoryManagerBuilder",
]
