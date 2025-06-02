from __future__ import annotations

"""
Memory Graph API module for Saplings.

This module provides the public API for memory graphs.
"""

from typing import Any, Dict, Optional

from saplings.api.stability import beta, stable
from saplings.memory._internal.dependency_graph import DependencyGraph as _DependencyGraph


@stable
class DependencyGraph(_DependencyGraph):
    """
    Dependency graph for memory.

    This class represents a dependency graph that stores relationships
    between documents and entities.
    """


@beta
class DependencyGraphBuilder:
    """
    Builder for dependency graphs.

    This class provides a fluent interface for building dependency graphs.
    """

    def __init__(self) -> None:
        """
        Initialize the builder.
        """
        self._graph = DependencyGraph()

    def add_node(
        self,
        node_id: str,
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DependencyGraphBuilder":
        """
        Add a node to the graph.

        Args:
        ----
            node_id: ID of the node
            node_type: Type of the node
            metadata: Node metadata

        Returns:
        -------
            Self for chaining

        """
        self._graph.add_node(node_id, node_type, metadata or {})
        return self

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DependencyGraphBuilder":
        """
        Add an edge to the graph.

        Args:
        ----
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Type of the edge
            metadata: Edge metadata

        Returns:
        -------
            Self for chaining

        """
        self._graph.add_edge(source_id, target_id, edge_type, metadata or {})
        return self

    def build(self) -> DependencyGraph:
        """
        Build the graph.

        Returns
        -------
            The built graph

        """
        return self._graph


__all__ = [
    "DependencyGraph",
    "DependencyGraphBuilder",
]
