"""
Graph module for Saplings memory.

This module defines the DependencyGraph class and related components.
"""

import json
import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from saplings.memory.config import MemoryConfig
from saplings.memory.document import Document
from saplings.memory.indexer import Entity, Relationship

logger = logging.getLogger(__name__)


class Node(ABC):
    """
    Base class for graph nodes.

    A node represents an entity in the dependency graph, such as a document
    or an extracted entity.
    """

    def __init__(self, id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a node.

        Args:
            id: Unique identifier for the node
            metadata: Additional metadata for the node
        """
        self.id = id
        self.metadata = metadata or {}

    def __eq__(self, other: object) -> bool:
        """
        Check if two nodes are equal.

        Args:
            other: Other node

        Returns:
            bool: True if the nodes are equal, False otherwise
        """
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """
        Get the hash of the node.

        Returns:
            int: Hash value
        """
        return hash(self.id)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the node to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "metadata": self.metadata,
        }


class DocumentNode(Node):
    """
    Node representing a document.

    A document node contains a reference to the document and additional metadata.
    """

    def __init__(
        self, document: Document, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a document node.

        Args:
            document: Document
            metadata: Additional metadata
        """
        super().__init__(id=document.id, metadata=metadata or {})
        self.document = document

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document node to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        data = super().to_dict()
        data["document_id"] = self.document.id
        return data


class EntityNode(Node):
    """
    Node representing an entity.

    An entity node contains information about an entity extracted from documents.
    """

    def __init__(
        self, entity: Entity, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an entity node.

        Args:
            entity: Entity
            metadata: Additional metadata
        """
        super().__init__(
            id=f"entity:{entity.entity_type}:{entity.name}",
            metadata=metadata or {},
        )
        self.entity = entity

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entity node to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        data = super().to_dict()
        data["entity"] = self.entity.to_dict()
        return data


class Edge:
    """
    Edge class for representing relationships between nodes.

    An edge connects two nodes with a specific relationship type and metadata.
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an edge.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Type of the relationship
            weight: Weight of the edge
            metadata: Additional metadata
        """
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.weight = weight
        self.metadata = metadata or {}

    def __eq__(self, other: object) -> bool:
        """
        Check if two edges are equal.

        Args:
            other: Other edge

        Returns:
            bool: True if the edges are equal, False otherwise
        """
        if not isinstance(other, Edge):
            return False
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.relationship_type == other.relationship_type
        )

    def __hash__(self) -> int:
        """
        Get the hash of the edge.

        Returns:
            int: Hash value
        """
        return hash((self.source_id, self.target_id, self.relationship_type))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the edge to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "metadata": self.metadata,
        }


class DependencyGraph:
    """
    Dependency graph for representing relationships between documents and entities.

    The dependency graph is a directed graph where nodes represent documents and entities,
    and edges represent relationships between them.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the dependency graph.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig.default()
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str, str], Edge] = {}

    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph.

        Args:
            node: Node to add
        """
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            self.graph.add_node(node.id, **node.metadata)
            logger.debug(f"Added node: {node.id}")

    def add_document_node(self, document: Document) -> DocumentNode:
        """
        Add a document node to the graph.

        Args:
            document: Document to add

        Returns:
            DocumentNode: Document node
        """
        node = DocumentNode(document=document)
        self.add_node(node)
        return node

    def add_entity_node(self, entity: Entity) -> EntityNode:
        """
        Add an entity node to the graph.

        Args:
            entity: Entity to add

        Returns:
            EntityNode: Entity node
        """
        node = EntityNode(entity=entity)
        self.add_node(node)
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Edge:
        """
        Add an edge to the graph.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Type of the relationship
            weight: Weight of the edge
            metadata: Additional metadata

        Returns:
            Edge: Edge

        Raises:
            ValueError: If the source or target node does not exist
        """
        if source_id not in self.nodes:
            raise ValueError(f"Source node not found: {source_id}")

        if target_id not in self.nodes:
            raise ValueError(f"Target node not found: {target_id}")

        edge_key = (source_id, target_id, relationship_type)

        if edge_key in self.edges:
            # Update existing edge
            edge = self.edges[edge_key]
            edge.weight = weight
            if metadata:
                edge.metadata.update(metadata)
        else:
            # Create new edge
            edge = Edge(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                weight=weight,
                metadata=metadata,
            )
            self.edges[edge_key] = edge

        # Add edge to NetworkX graph
        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship_type,
            weight=weight,
            **edge.metadata,
        )

        logger.debug(f"Added edge: {source_id} --[{relationship_type}]--> {target_id}")
        return edge

    def add_relationship(self, relationship: Relationship) -> Edge:
        """
        Add a relationship to the graph.

        Args:
            relationship: Relationship to add

        Returns:
            Edge: Edge

        Raises:
            ValueError: If the source or target node does not exist
        """
        return self.add_edge(
            source_id=relationship.source_id,
            target_id=relationship.target_id,
            relationship_type=relationship.relationship_type,
            weight=relationship.metadata.get("confidence", 1.0),
            metadata=relationship.metadata,
        )

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get a node by ID.

        Args:
            node_id: ID of the node

        Returns:
            Optional[Node]: Node if found, None otherwise
        """
        return self.nodes.get(node_id)

    def get_edge(
        self, source_id: str, target_id: str, relationship_type: str
    ) -> Optional[Edge]:
        """
        Get an edge by source, target, and relationship type.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Type of the relationship

        Returns:
            Optional[Edge]: Edge if found, None otherwise
        """
        edge_key = (source_id, target_id, relationship_type)
        return self.edges.get(edge_key)

    def get_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "outgoing",
    ) -> List[Node]:
        """
        Get neighbors of a node.

        Args:
            node_id: ID of the node
            relationship_types: Types of relationships to consider
            direction: Direction of the edges ("outgoing", "incoming", or "both")

        Returns:
            List[Node]: Neighbor nodes

        Raises:
            ValueError: If the node does not exist or the direction is invalid
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")

        if direction not in ["outgoing", "incoming", "both"]:
            raise ValueError(f"Invalid direction: {direction}")

        neighbors = []

        if direction == "outgoing" or direction == "both":
            for _, target_id, data in self.graph.out_edges(node_id, data=True):
                if relationship_types is None or data["relationship_type"] in relationship_types:
                    neighbor = self.nodes.get(target_id)
                    if neighbor:
                        neighbors.append(neighbor)

        if direction == "incoming" or direction == "both":
            for source_id, _, data in self.graph.in_edges(node_id, data=True):
                if relationship_types is None or data["relationship_type"] in relationship_types:
                    neighbor = self.nodes.get(source_id)
                    if neighbor:
                        neighbors.append(neighbor)

        return neighbors

    def get_subgraph(
        self,
        node_ids: List[str],
        max_hops: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> "DependencyGraph":
        """
        Get a subgraph centered on the specified nodes.

        Args:
            node_ids: IDs of the central nodes
            max_hops: Maximum number of hops from the central nodes
            relationship_types: Types of relationships to consider

        Returns:
            DependencyGraph: Subgraph

        Raises:
            ValueError: If any of the nodes does not exist
        """
        for node_id in node_ids:
            if node_id not in self.nodes:
                raise ValueError(f"Node not found: {node_id}")

        # Create a new graph with the same configuration
        subgraph = DependencyGraph(config=self.config)

        # Add the central nodes
        for node_id in node_ids:
            subgraph.add_node(self.nodes[node_id])

        # Add nodes and edges within max_hops
        frontier = set(node_ids)
        visited = set(node_ids)

        for _ in range(max_hops):
            new_frontier = set()

            for node_id in frontier:
                # Get outgoing edges
                for _, target_id, data in self.graph.out_edges(node_id, data=True):
                    if relationship_types is None or data["relationship_type"] in relationship_types:
                        if target_id not in visited:
                            subgraph.add_node(self.nodes[target_id])
                            new_frontier.add(target_id)

                        edge_key = (node_id, target_id, data["relationship_type"])
                        if edge_key in self.edges:
                            subgraph.add_edge(
                                source_id=node_id,
                                target_id=target_id,
                                relationship_type=data["relationship_type"],
                                weight=data.get("weight", 1.0),
                                metadata={k: v for k, v in data.items() if k not in ["relationship_type", "weight"]},
                            )

                # Get incoming edges
                for source_id, _, data in self.graph.in_edges(node_id, data=True):
                    if relationship_types is None or data["relationship_type"] in relationship_types:
                        if source_id not in visited:
                            subgraph.add_node(self.nodes[source_id])
                            new_frontier.add(source_id)

                        edge_key = (source_id, node_id, data["relationship_type"])
                        if edge_key in self.edges:
                            subgraph.add_edge(
                                source_id=source_id,
                                target_id=node_id,
                                relationship_type=data["relationship_type"],
                                weight=data.get("weight", 1.0),
                                metadata={k: v for k, v in data.items() if k not in ["relationship_type", "weight"]},
                            )

            visited.update(new_frontier)
            frontier = new_frontier

            if not frontier:
                break

        return subgraph

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        relationship_types: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find paths between two nodes.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            max_hops: Maximum number of hops
            relationship_types: Types of relationships to consider

        Returns:
            List[List[Tuple[str, str, str]]]: List of paths, where each path is a list of
                (source_id, relationship_type, target_id) tuples

        Raises:
            ValueError: If the source or target node does not exist
        """
        if source_id not in self.nodes:
            raise ValueError(f"Source node not found: {source_id}")

        if target_id not in self.nodes:
            raise ValueError(f"Target node not found: {target_id}")

        # Create a filtered graph if relationship_types is specified
        if relationship_types:
            filtered_graph = nx.DiGraph()

            for u, v, data in self.graph.edges(data=True):
                if data["relationship_type"] in relationship_types:
                    filtered_graph.add_edge(u, v, **data)

            graph = filtered_graph
        else:
            graph = self.graph

        # Find all simple paths
        paths = []
        for path in nx.all_simple_paths(graph, source_id, target_id, cutoff=max_hops):
            edge_path = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = graph.get_edge_data(u, v)

                for rel_type, data in edge_data.items():
                    if rel_type != "relationship_type":
                        continue

                    edge_path.append((u, data, v))

            paths.append(edge_path)

        return paths

    def get_connected_components(
        self, relationship_types: Optional[List[str]] = None
    ) -> List[Set[str]]:
        """
        Get connected components of the graph.

        Args:
            relationship_types: Types of relationships to consider

        Returns:
            List[Set[str]]: List of connected components, where each component is a set of node IDs
        """
        # Create an undirected graph for connected components
        undirected = nx.Graph()

        for u, v, data in self.graph.edges(data=True):
            if relationship_types is None or data["relationship_type"] in relationship_types:
                undirected.add_edge(u, v)

        return [set(component) for component in nx.connected_components(undirected)]

    def get_centrality(
        self, centrality_type: str = "degree", relationship_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate centrality measures for nodes.

        Args:
            centrality_type: Type of centrality ("degree", "betweenness", "closeness", or "eigenvector")
            relationship_types: Types of relationships to consider

        Returns:
            Dict[str, float]: Dictionary mapping node IDs to centrality values

        Raises:
            ValueError: If the centrality type is invalid
        """
        if centrality_type not in ["degree", "betweenness", "closeness", "eigenvector"]:
            raise ValueError(f"Invalid centrality type: {centrality_type}")

        # Create a filtered graph if relationship_types is specified
        if relationship_types:
            filtered_graph = nx.DiGraph()

            for u, v, data in self.graph.edges(data=True):
                if data["relationship_type"] in relationship_types:
                    filtered_graph.add_edge(u, v, **data)

            graph = filtered_graph
        else:
            graph = self.graph

        # Calculate centrality
        if centrality_type == "degree":
            return dict(graph.degree())
        elif centrality_type == "betweenness":
            return nx.betweenness_centrality(graph)
        elif centrality_type == "closeness":
            return nx.closeness_centrality(graph)
        elif centrality_type == "eigenvector":
            return nx.eigenvector_centrality(graph, max_iter=1000)

    def save(self, directory: str) -> None:
        """
        Save the graph to disk.

        Args:
            directory: Directory to save to
        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save nodes
        nodes_data = {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        with open(directory_path / "nodes.json", "w") as f:
            json.dump(nodes_data, f)

        # Save edges
        edges_data = {
            f"{source_id}|{target_id}|{rel_type}": edge.to_dict()
            for (source_id, target_id, rel_type), edge in self.edges.items()
        }
        with open(directory_path / "edges.json", "w") as f:
            json.dump(edges_data, f)

        # Save graph structure
        import pickle
        with open(directory_path / "graph.gpickle", "wb") as f:
            pickle.dump(self.graph, f)

        logger.info(f"Saved graph to {directory}")

    def load(self, directory: str) -> None:
        """
        Load the graph from disk.

        Args:
            directory: Directory to load from

        Raises:
            FileNotFoundError: If the graph files are not found
        """
        directory_path = Path(directory)

        # Check if files exist
        nodes_path = directory_path / "nodes.json"
        edges_path = directory_path / "edges.json"
        graph_path = directory_path / "graph.gpickle"

        if not nodes_path.exists() or not edges_path.exists() or not graph_path.exists():
            raise FileNotFoundError(f"Graph files not found in {directory}")

        # Clear existing data
        self.nodes.clear()
        self.edges.clear()
        self.graph = nx.DiGraph()

        # Load graph structure
        import pickle
        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)

        # Load nodes
        with open(nodes_path, "r") as f:
            nodes_data = json.load(f)

        for node_id, node_data in nodes_data.items():
            node_type = node_data["type"]

            if node_type == "DocumentNode":
                # We need to load the document separately
                document_id = node_data["document_id"]
                # For now, create a placeholder node
                node = Node(id=node_id, metadata=node_data["metadata"])
                self.nodes[node_id] = node

            elif node_type == "EntityNode":
                # Create an entity from the data
                entity_data = node_data["entity"]
                entity = Entity(
                    name=entity_data["name"],
                    entity_type=entity_data["entity_type"],
                    metadata=entity_data.get("metadata", {}),
                )
                node = EntityNode(entity=entity, metadata=node_data["metadata"])
                self.nodes[node_id] = node

            else:
                # Generic node
                node = Node(id=node_id, metadata=node_data["metadata"])
                self.nodes[node_id] = node

        # Load edges
        with open(edges_path, "r") as f:
            edges_data = json.load(f)

        for edge_key, edge_data in edges_data.items():
            source_id, target_id, rel_type = edge_key.split("|")
            edge = Edge(
                source_id=source_id,
                target_id=target_id,
                relationship_type=rel_type,
                weight=edge_data["weight"],
                metadata=edge_data["metadata"],
            )
            self.edges[(source_id, target_id, rel_type)] = edge

        logger.info(f"Loaded graph from {directory}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the graph to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {
                f"{source_id}|{target_id}|{rel_type}": edge.to_dict()
                for (source_id, target_id, rel_type), edge in self.edges.items()
            },
        }

    def __len__(self) -> int:
        """
        Get the number of nodes in the graph.

        Returns:
            int: Number of nodes
        """
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        """
        Check if a node is in the graph.

        Args:
            node_id: ID of the node

        Returns:
            bool: True if the node is in the graph, False otherwise
        """
        return node_id in self.nodes
