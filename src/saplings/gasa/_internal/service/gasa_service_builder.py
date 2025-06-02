from __future__ import annotations

"""
GASA Service Builder module for Saplings.

This module provides a builder class for creating GASAService instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.
"""

import logging
from typing import Any, Callable, Protocol, runtime_checkable

from saplings.core._internal.builder import ServiceBuilder
from saplings.gasa._internal.config import GASAConfig
from saplings.gasa._internal.service.gasa_service import GASAService

logger = logging.getLogger(__name__)


@runtime_checkable
class DependencyGraph(Protocol):
    """Protocol for dependency graph objects."""

    def get_neighbors(self, node_id: str) -> list[str]: ...
    def get_distance(self, source_id: str, target_id: str) -> int | float: ...
    def get_subgraph(self, node_ids: list[str], max_hops: int = 2) -> "DependencyGraph": ...
    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None: ...
    def add_edge(
        self, source_id: str, target_id: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


class GASAServiceBuilder(ServiceBuilder[GASAService]):
    """
    Builder for GASAService.

    This class provides a fluent interface for building GASAService instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for GASAService
    builder = GASAServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_graph(dependency_graph) \
                    .with_config(gasa_config) \
                    .with_tokenizer(tokenizer) \
                    .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the GASA service builder."""
        super().__init__(GASAService)
        # We don't need to store these as instance variables since we're using the dependency system

    def with_graph(self, graph: Any) -> GASAServiceBuilder:
        """
        Set the dependency graph.

        Args:
        ----
            graph: Dependency graph for GASA

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("graph", graph)
        # We don't need to clear the provider since the direct dependency takes precedence
        return self

    def with_graph_provider(self, provider: Callable[[], Any]) -> GASAServiceBuilder:
        """
        Set a provider function for lazy loading of the dependency graph.

        This allows the GASA service to initialize without requiring the graph
        to be available immediately. The graph will be loaded on-demand when needed.

        Args:
        ----
            provider: Function that returns a dependency graph when called

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("graph_provider", provider)
        return self

    def with_config(self, config: GASAConfig) -> GASAServiceBuilder:
        """
        Set the GASA configuration.

        Args:
        ----
            config: GASA configuration

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("config", config)
        return self

    def with_tokenizer(self, tokenizer: Any) -> GASAServiceBuilder:
        """
        Set the tokenizer.

        Args:
        ----
            tokenizer: Tokenizer for converting text to tokens

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("tokenizer", tokenizer)
        return self

    def with_tokenizer_provider(self, provider: Callable[[], Any]) -> GASAServiceBuilder:
        """
        Set a provider function for lazy loading of the tokenizer.

        This allows the GASA service to initialize without requiring the tokenizer
        to be available immediately. The tokenizer will be loaded on-demand when needed.

        Args:
        ----
            provider: Function that returns a tokenizer when called

        Returns:
        -------
            The builder instance for method chaining

        """
        self.with_dependency("tokenizer_provider", provider)
        return self

    def build(self) -> GASAService:
        """
        Build the GASA service instance with the configured dependencies.

        The service will be initialized with the provided dependencies or providers,
        allowing for lazy loading of resources when needed.

        Returns
        -------
            The initialized GASA service instance

        """
        logger.debug("Building GASA service with lazy initialization support")
        return super().build()
