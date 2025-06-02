from __future__ import annotations

"""
Memory Store Builder module for Saplings.

This module provides a builder class for creating MemoryStore instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.exceptions import InitializationError
from saplings.memory._internal.config import MemoryConfig
from saplings.memory._internal.graph import DependencyGraph
from saplings.memory._internal.indexer_registry import IndexerRegistry, get_indexer
from saplings.memory._internal.memory_store import MemoryStore
from saplings.memory._internal.vector_store import get_vector_store

if TYPE_CHECKING:
    from saplings.core._internal.plugin import PluginRegistry

logger = logging.getLogger(__name__)


class MemoryStoreBuilder(ServiceBuilder[MemoryStore]):
    """
    Builder for MemoryStore.

    This class provides a fluent interface for building MemoryStore instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for MemoryStore
    builder = MemoryStoreBuilder()

    # Configure the builder with dependencies and options
    store = builder.with_config(memory_config) \
                  .with_vector_store(vector_store) \
                  .with_graph(graph) \
                  .build()

    # Or load from a directory
    store = builder.from_directory("./memory_store") \
                  .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the memory store builder."""
        super().__init__(MemoryStore)
        self._config: Optional[MemoryConfig] = None
        self._vector_store: Any = None
        self._graph: Optional[DependencyGraph] = None
        self._indexer: Any = None
        self._from_directory: Optional[str] = None
        self._indexer_registry: Optional[IndexerRegistry] = None
        self._plugin_registry: Optional["PluginRegistry"] = None

    def with_config(self, config: MemoryConfig) -> MemoryStoreBuilder:
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

    def with_vector_store(self, vector_store: Any) -> MemoryStoreBuilder:
        """
        Set the vector store.

        Args:
        ----
            vector_store: Vector store for document storage

        Returns:
        -------
            The builder instance for method chaining

        """
        self._vector_store = vector_store
        self.with_dependency("vector_store", vector_store)
        return self

    def with_graph(self, graph: DependencyGraph) -> MemoryStoreBuilder:
        """
        Set the dependency graph.

        Args:
        ----
            graph: Dependency graph for document relationships

        Returns:
        -------
            The builder instance for method chaining

        """
        self._graph = graph
        self.with_dependency("graph", graph)
        return self

    def with_indexer(self, indexer: Any) -> MemoryStoreBuilder:
        """
        Set the indexer.

        Args:
        ----
            indexer: Indexer for document indexing

        Returns:
        -------
            The builder instance for method chaining

        """
        self._indexer = indexer
        self.with_dependency("indexer", indexer)
        return self

    def with_indexer_registry(self, registry: IndexerRegistry) -> MemoryStoreBuilder:
        """
        Set the indexer registry.

        Args:
        ----
            registry: Indexer registry to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self._indexer_registry = registry
        self.with_dependency("indexer_registry", registry)
        return self

    def with_plugin_registry(self, registry: "PluginRegistry") -> MemoryStoreBuilder:
        """
        Set the plugin registry.

        Args:
        ----
            registry: Plugin registry to use

        Returns:
        -------
            The builder instance for method chaining

        """
        self._plugin_registry = registry
        self.with_dependency("plugin_registry", registry)
        return self

    def from_directory(self, directory: str) -> MemoryStoreBuilder:
        """
        Load the memory store from a directory.

        Args:
        ----
            directory: Directory to load from

        Returns:
        -------
            The builder instance for method chaining

        """
        self._from_directory = directory
        return self

    def build(self) -> MemoryStore:
        """
        Build the memory store instance with the configured dependencies.

        Returns
        -------
            The initialized memory store instance

        Raises
        ------
            InitializationError: If initialization fails

        """
        try:
            # If loading from directory, create a basic memory store first
            if self._from_directory:
                # Ensure the directory exists
                if not os.path.exists(self._from_directory):
                    raise InitializationError(f"Directory {self._from_directory} does not exist")

                # Create a basic memory store with default config
                if not self._config:
                    self._config = MemoryConfig.default()
                    self.with_dependency("config", self._config)

                # Create the memory store directly
                memory_store = MemoryStore(
                    config=self._config,
                    indexer_registry=self._indexer_registry,
                    plugin_registry=self._plugin_registry,
                    vector_store=self._vector_store,
                    graph=self._graph,
                    indexer=self._indexer,
                )

                # Load from directory
                memory_store.load(self._from_directory)
                return memory_store

            # Otherwise, build with provided dependencies
            if not self._config:
                self._config = MemoryConfig.default()
                self.with_dependency("config", self._config)

            # Create vector store if not provided
            if not self._vector_store:
                self._vector_store = get_vector_store(
                    config=self._config, registry=self._plugin_registry
                )
                self.with_dependency("vector_store", self._vector_store)

            # Create graph if not provided
            if not self._graph:
                self._graph = DependencyGraph(config=self._config)
                self.with_dependency("graph", self._graph)

            # Create indexer if not provided
            if not self._indexer:
                self._indexer = get_indexer(config=self._config, registry=self._indexer_registry)
                self.with_dependency("indexer", self._indexer)

            # Create the memory store directly instead of using super().build()
            return MemoryStore(
                config=self._config,
                indexer_registry=self._indexer_registry,
                plugin_registry=self._plugin_registry,
                vector_store=self._vector_store,
                graph=self._graph,
                indexer=self._indexer,
            )
        except Exception as e:
            if isinstance(e, InitializationError):
                raise
            raise InitializationError(f"Failed to initialize MemoryStore: {e}", cause=e)
