from __future__ import annotations

"""
Dependency graph module for Saplings memory.

This module defines the dependency graph for memory.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning(
        "NetworkX not installed. Graph functionality will be limited. "
        "Install NetworkX with: pip install networkx"
    )


class DependencyGraph:
    """
    Dependency graph for memory.

    This class represents a dependency graph that stores relationships
    between documents and entities.
    """

    def __init__(self) -> None:
        """
        Initialize the dependency graph.
        """
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Initialize NetworkX graph if available
        self.graph = None
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()

    def add_node(
        self,
        node_id: str,
        node_type: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add a node to the graph.

        Args:
        ----
            node_id: ID of the node
            node_type: Type of the node
            metadata: Node metadata

        """
        # Create node data
        node_data = {
            "id": node_id,
            "type": node_type,
            "metadata": metadata,
        }

        # Add to internal dictionary
        self.nodes[node_id] = node_data

        # Add to NetworkX graph if available
        if self.graph is not None:
            self.graph.add_node(node_id, **node_data)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add an edge to the graph.

        Args:
        ----
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Type of the edge
            metadata: Edge metadata

        """
        # Create edge data
        edge_data = {
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "metadata": metadata,
        }

        # Add to internal dictionary
        self.edges[(source_id, target_id)] = edge_data

        # Add to NetworkX graph if available
        if self.graph is not None:
            self.graph.add_edge(source_id, target_id, **edge_data)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID.

        Args:
        ----
            node_id: ID of the node

        Returns:
        -------
            Node data or None if not found

        """
        return self.nodes.get(node_id)

    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an edge by source and target IDs.

        Args:
        ----
            source_id: ID of the source node
            target_id: ID of the target node

        Returns:
        -------
            Edge data or None if not found

        """
        return self.edges.get((source_id, target_id))

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """
        Get neighbors of a node.

        Args:
        ----
            node_id: ID of the node
            edge_type: Type of edges to consider (optional)

        Returns:
        -------
            List of neighbor node IDs

        """
        neighbors = []

        # Check outgoing edges
        for (source, target), edge in self.edges.items():
            if source == node_id:
                if edge_type is None or edge["type"] == edge_type:
                    neighbors.append(target)

        return neighbors

    def get_predecessors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """
        Get predecessors of a node.

        Args:
        ----
            node_id: ID of the node
            edge_type: Type of edges to consider (optional)

        Returns:
        -------
            List of predecessor node IDs

        """
        predecessors = []

        # Check incoming edges
        for (source, target), edge in self.edges.items():
            if target == node_id:
                if edge_type is None or edge["type"] == edge_type:
                    predecessors.append(source)

        return predecessors

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """
        Get nodes by type.

        Args:
        ----
            node_type: Type of nodes to get

        Returns:
        -------
            List of node data

        """
        return [node for node in self.nodes.values() if node["type"] == node_type]

    def get_edges_by_type(self, edge_type: str) -> List[Dict[str, Any]]:
        """
        Get edges by type.

        Args:
        ----
            edge_type: Type of edges to get

        Returns:
        -------
            List of edge data

        """
        return [edge for edge in self.edges.values() if edge["type"] == edge_type]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the graph to a dictionary.

        Returns
        -------
            Dictionary representation

        """
        return {
            "nodes": list(self.nodes.values()),
            "edges": list(self.edges.values()),
        }

    def to_json(self) -> str:
        """
        Convert the graph to JSON.

        Returns
        -------
            JSON representation

        """
        return json.dumps(self.to_dict())

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load the graph from a dictionary.

        Args:
        ----
            data: Dictionary representation

        """
        # Clear existing data
        self.nodes.clear()
        self.edges.clear()
        if self.graph is not None:
            self.graph.clear()

        # Load nodes
        for node_data in data.get("nodes", []):
            node_id = node_data["id"]
            node_type = node_data["type"]
            metadata = node_data.get("metadata", {})
            self.add_node(node_id, node_type, metadata)

        # Load edges
        for edge_data in data.get("edges", []):
            source_id = edge_data["source"]
            target_id = edge_data["target"]
            edge_type = edge_data["type"]
            metadata = edge_data.get("metadata", {})
            self.add_edge(source_id, target_id, edge_type, metadata)

    def from_json(self, json_str: str) -> None:
        """
        Load the graph from JSON.

        Args:
        ----
            json_str: JSON representation

        """
        data = json.loads(json_str)
        self.from_dict(data)

    def merge(self, other: "DependencyGraph") -> None:
        """
        Merge another graph into this one.

        Args:
        ----
            other: Graph to merge

        """
        # Merge nodes
        for node_id, node_data in other.nodes.items():
            if node_id not in self.nodes:
                self.add_node(
                    node_id,
                    node_data["type"],
                    node_data.get("metadata", {}),
                )
            else:
                # Update metadata
                self.nodes[node_id]["metadata"].update(node_data.get("metadata", {}))

        # Merge edges
        for (source_id, target_id), edge_data in other.edges.items():
            if (source_id, target_id) not in self.edges:
                self.add_edge(
                    source_id,
                    target_id,
                    edge_data["type"],
                    edge_data.get("metadata", {}),
                )
            else:
                # Update metadata
                self.edges[(source_id, target_id)]["metadata"].update(edge_data.get("metadata", {}))

    def get_subgraph(self, node_ids: List[str]) -> "DependencyGraph":
        """
        Get a subgraph containing only the specified nodes.

        Args:
        ----
            node_ids: IDs of nodes to include

        Returns:
        -------
            Subgraph

        """
        subgraph = DependencyGraph()

        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                node_data = self.nodes[node_id]
                subgraph.add_node(
                    node_id,
                    node_data["type"],
                    node_data.get("metadata", {}),
                )

        # Add edges
        for (source_id, target_id), edge_data in self.edges.items():
            if source_id in node_ids and target_id in node_ids:
                subgraph.add_edge(
                    source_id,
                    target_id,
                    edge_data["type"],
                    edge_data.get("metadata", {}),
                )

        return subgraph
