from __future__ import annotations

"""
Dependency Graph Builder module for Saplings.

This module provides a builder class for creating DependencyGraph instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
import os
from typing import TYPE_CHECKING

from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.exceptions import InitializationError
from saplings.memory._internal.config import MemoryConfig
from saplings.memory._internal.graph import DependencyGraph

if TYPE_CHECKING:
    from saplings.memory._internal.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class DependencyGraphBuilder(ServiceBuilder[DependencyGraph]):
    """
    Builder for DependencyGraph.

    This class provides a fluent interface for building DependencyGraph instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for DependencyGraph
    builder = DependencyGraphBuilder()

    # Configure the builder with dependencies and options
    graph = builder.with_config(memory_config) \
                  .build()

    # Or load from a directory
    graph = builder.from_directory("./graph") \
                  .build()

    # Or build from a memory store
    graph = builder.from_memory_store(memory_store) \
                  .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the dependency graph builder."""
        super().__init__(DependencyGraph)
        self._config = None
        self._from_directory = None
        self._from_memory_store = None

    def with_config(self, config: MemoryConfig) -> DependencyGraphBuilder:
        """
        Set the memory configuration.

        Args:
        ----
            config: Memory configuration

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config = config
        self.with_dependency("config", config)
        return self

    def from_directory(self, directory: str) -> DependencyGraphBuilder:
        """
        Load the dependency graph from a directory.

        Args:
        ----
            directory: Directory to load from

        Returns:
        -------
            The builder instance for method chaining

        """
        self._from_directory = directory
        return self

    def from_memory_store(self, memory_store: "MemoryStore") -> DependencyGraphBuilder:
        """
        Build the dependency graph from a memory store.

        Args:
        ----
            memory_store: Memory store to build from

        Returns:
        -------
            The builder instance for method chaining

        """
        self._from_memory_store = memory_store
        return self

    def build(self) -> DependencyGraph:
        """
        Build the dependency graph instance with the configured dependencies.

        Returns
        -------
            The initialized dependency graph instance

        Raises
        ------
            InitializationError: If initialization fails

        """
        try:
            # If no config provided, use default
            if not self._config:
                self._config = MemoryConfig.default()
                self.with_dependency("config", self._config)

            # Build the dependency graph
            graph = super().build()

            # If loading from directory, load the graph
            if self._from_directory:
                # Ensure the directory exists
                if not os.path.exists(self._from_directory):
                    raise InitializationError(f"Directory {self._from_directory} does not exist")

                # Load from directory
                graph.load(self._from_directory)

            # If building from memory store, build the graph
            if self._from_memory_store:
                # Build from memory store
                graph.build_from_memory(self._from_memory_store)

            return graph
        except Exception as e:
            if isinstance(e, InitializationError):
                raise
            raise InitializationError(f"Failed to initialize DependencyGraph: {e}", cause=e)
