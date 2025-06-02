from __future__ import annotations

"""
Graph module for Saplings memory.

This module defines the DependencyGraph class and related components.
"""


import json
import logging
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from saplings.memory._internal.config import MemoryConfig
from saplings.memory._internal.entity import Entity
from saplings.memory._internal.relationship import Relationship

if TYPE_CHECKING:
    from saplings.memory._internal.document import Document
    # Use forward reference for MemoryStore to avoid circular dependency
    # Don't import from saplings.api.memory to avoid circular imports
    # Define MemoryStore as a string type hint instead

logger = logging.getLogger(__name__)


def _get_source_from_metadata(metadata: Any) -> str | None:
    """
    Safely extract source from document metadata.

    Args:
    ----
        metadata: Document metadata (can be dict, DocumentMetadata, or None)

    Returns:
    -------
        Optional[str]: Source if found, None otherwise

    """
    if metadata is None:
        return None

    if isinstance(metadata, dict) and "source" in metadata:
        return metadata["source"]

    # Check if metadata has a source attribute
    if hasattr(metadata, "source"):
        # Use getattr with a default to handle None values
        return getattr(metadata, "source", None)

    return None


class Node(ABC):
    """
    Base class for graph nodes.

    A node represents an entity in the dependency graph, such as a document
    or an extracted entity.
    """

    def __init__(self, id: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Initialize a node.

        Args:
        ----
            id: Unique identifier for the node
            metadata: Additional metadata for the node

        """
        self.id = id
        self.metadata = metadata or {}

    def __eq__(self, other: object) -> bool:
        """
        Check if two nodes are equal.

        Args:
        ----
            other: Other node

        Returns:
        -------
            bool: True if the nodes are equal, False otherwise

        """
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self):
        """
        Get the hash of the node.

        Returns
        -------
            int: Hash value

        """
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the node to a dictionary.

        Returns
        -------
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

    def __init__(self, document: Document, metadata: dict[str, Any] | None = None) -> None:
        """
        Initialize a document node.

        Args:
        ----
            document: Document
            metadata: Additional metadata

        """
        super().__init__(id=document.id, metadata=metadata or {})
        self.document = document

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the document node to a dictionary.

        Returns
        -------
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

    def __init__(self, entity: Entity, metadata: dict[str, Any] | None = None) -> None:
        """
        Initialize an entity node.

        Args:
        ----
            entity: Entity
            metadata: Additional metadata

        """
        super().__init__(
            id=f"entity:{entity.entity_type}:{entity.name}",
            metadata=metadata or {},
        )
        self.entity = entity

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the entity node to a dictionary.

        Returns
        -------
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize an edge.

        Args:
        ----
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
        ----
            other: Other edge

        Returns:
        -------
            bool: True if the edges are equal, False otherwise

        """
        if not isinstance(other, Edge):
            return False
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.relationship_type == other.relationship_type
        )

    def __hash__(self):
        """
        Get the hash of the edge.

        Returns
        -------
            int: Hash value

        """
        return hash((self.source_id, self.target_id, self.relationship_type))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the edge to a dictionary.

        Returns
        -------
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

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """
        Initialize the dependency graph.

        Args:
        ----
            config: Memory configuration

        """
        self.config = config or MemoryConfig.default()
        self.graph = nx.DiGraph()
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str, str], Edge] = {}

    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph.

        Args:
        ----
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
        ----
            document: Document to add

        Returns:
        -------
            DocumentNode: Document node

        """
        node = DocumentNode(document=document)
        self.add_node(node)
        return node

    def add_entity_node(self, entity: Entity) -> EntityNode:
        """
        Add an entity node to the graph.

        Args:
        ----
            entity: Entity to add

        Returns:
        -------
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
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        """
        Add an edge to the graph.

        Args:
        ----
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Type of the relationship
            weight: Weight of the edge
            metadata: Additional metadata

        Returns:
        -------
            Edge: Edge

        Raises:
        ------
            ValueError: If the source or target node does not exist

        """
        if source_id not in self.nodes:
            msg = f"Source node not found: {source_id}"
            raise ValueError(msg)

        if target_id not in self.nodes:
            msg = f"Target node not found: {target_id}"
            raise ValueError(msg)

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

    def add_relationship(
        self,
        source_id_or_relationship: Relationship | str,
        target_id: str | None = None,
        relationship_type: str | None = None,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        """
        Add a relationship to the graph.

        This method can be called in two ways:
        1. With a Relationship object: add_relationship(relationship)
        2. With source_id, target_id, and relationship_type: add_relationship(source_id, target_id, relationship_type, weight=1.0, metadata=None)

        Args:
        ----
            source_id_or_relationship: Either a Relationship object or the source node ID
            target_id: Target node ID (if source_id_or_relationship is a string)
            relationship_type: Type of the relationship (if source_id_or_relationship is a string)
            weight: Weight of the edge (if source_id_or_relationship is a string)
            metadata: Additional metadata (if source_id_or_relationship is a string)

        Returns:
        -------
            Edge: Edge

        Raises:
        ------
            ValueError: If the source or target node does not exist or if required parameters are missing

        """
        if isinstance(source_id_or_relationship, Relationship):
            # Called with a Relationship object
            relationship = source_id_or_relationship
            return self.add_edge(
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                relationship_type=relationship.relationship_type,
                weight=relationship.metadata.get("confidence", 1.0),
                metadata=relationship.metadata,
            )
        # Called with source_id, target_id, and relationship_type
        if target_id is None or relationship_type is None:
            msg = "When calling add_relationship with a source_id, both target_id and relationship_type must be provided"
            raise ValueError(msg)

        return self.add_edge(
            source_id=source_id_or_relationship,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            metadata=metadata,
        )

    def get_node(self, node_id: str) -> Node | None:
        """
        Get a node by ID.

        Args:
        ----
            node_id: ID of the node

        Returns:
        -------
            Optional[Node]: Node if found, None otherwise

        """
        return self.nodes.get(node_id)

    def get_edge(self, source_id: str, target_id: str, relationship_type: str) -> Edge | None:
        """
        Get an edge by source, target, and relationship type.

        Args:
        ----
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Type of the relationship

        Returns:
        -------
            Optional[Edge]: Edge if found, None otherwise

        """
        edge_key = (source_id, target_id, relationship_type)
        return self.edges.get(edge_key)

    def get_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None = None,
        direction: str = "outgoing",
    ) -> list[Node]:
        """
        Get neighbors of a node.

        Args:
        ----
            node_id: ID of the node
            relationship_types: Types of relationships to consider
            direction: Direction of the edges ("outgoing", "incoming", or "both")

        Returns:
        -------
            List[Node]: Neighbor nodes

        Raises:
        ------
            ValueError: If the node does not exist or the direction is invalid

        """
        if node_id not in self.nodes:
            msg = f"Node not found: {node_id}"
            raise ValueError(msg)

        if direction not in ["outgoing", "incoming", "both"]:
            msg = f"Invalid direction: {direction}"
            raise ValueError(msg)

        neighbors = []

        if direction in {"outgoing", "both"}:
            for _, target_id, data in self.graph.out_edges(node_id, data=True):
                if relationship_types is None or data["relationship_type"] in relationship_types:
                    neighbor = self.nodes.get(target_id)
                    if neighbor:
                        neighbors.append(neighbor)

        if direction in {"incoming", "both"}:
            for source_id, _, data in self.graph.in_edges(node_id, data=True):
                if relationship_types is None or data["relationship_type"] in relationship_types:
                    neighbor = self.nodes.get(source_id)
                    if neighbor:
                        neighbors.append(neighbor)

        return neighbors

    def get_subgraph(
        self,
        node_ids: list[str],
        max_hops: int = 1,
        relationship_types: list[str] | None = None,
    ) -> "DependencyGraph":
        """
        Get a subgraph centered on the specified nodes.

        Args:
        ----
            node_ids: IDs of the central nodes
            max_hops: Maximum number of hops from the central nodes
            relationship_types: Types of relationships to consider

        Returns:
        -------
            DependencyGraph: Subgraph

        Raises:
        ------
            ValueError: If any of the nodes does not exist

        """
        for node_id in node_ids:
            if node_id not in self.nodes:
                msg = f"Node not found: {node_id}"
                raise ValueError(msg)

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
                    if (
                        relationship_types is None
                        or data["relationship_type"] in relationship_types
                    ):
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
                                metadata={
                                    k: v
                                    for k, v in data.items()
                                    if k not in ["relationship_type", "weight"]
                                },
                            )

                # Get incoming edges
                for source_id, _, data in self.graph.in_edges(node_id, data=True):
                    if (
                        relationship_types is None
                        or data["relationship_type"] in relationship_types
                    ):
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
                                metadata={
                                    k: v
                                    for k, v in data.items()
                                    if k not in ["relationship_type", "weight"]
                                },
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
        relationship_types: list[str] | None = None,
    ) -> list[list[tuple[str, str, str]]]:
        """
        Find paths between two nodes.

        Args:
        ----
            source_id: ID of the source node
            target_id: ID of the target node
            max_hops: Maximum number of hops
            relationship_types: Types of relationships to consider

        Returns:
        -------
            List[List[Tuple[str, str, str]]]: List of paths, where each path is a list of
                (source_id, relationship_type, target_id) tuples

        Raises:
        ------
            ValueError: If the source or target node does not exist

        """
        if source_id not in self.nodes:
            msg = f"Source node not found: {source_id}"
            raise ValueError(msg)

        if target_id not in self.nodes:
            msg = f"Target node not found: {target_id}"
            raise ValueError(msg)

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
        self, relationship_types: list[str] | None = None
    ) -> list[set[str]]:
        """
        Get connected components of the graph.

        Args:
        ----
            relationship_types: Types of relationships to consider

        Returns:
        -------
            List[Set[str]]: List of connected components, where each component is a set of node IDs

        """
        # Create an undirected graph for connected components
        undirected = nx.Graph()

        for u, v, data in self.graph.edges(data=True):
            if relationship_types is None or data["relationship_type"] in relationship_types:
                undirected.add_edge(u, v)

        return [set(component) for component in nx.connected_components(undirected)]

    def get_centrality(
        self, centrality_type: str = "degree", relationship_types: list[str] | None = None
    ) -> dict[str, float]:
        """
        Calculate centrality measures for nodes.

        Args:
        ----
            centrality_type: Type of centrality ("degree", "betweenness", "closeness", or "eigenvector")
            relationship_types: Types of relationships to consider

        Returns:
        -------
            Dict[str, float]: Dictionary mapping node IDs to centrality values

        Raises:
        ------
            ValueError: If the centrality type is invalid

        """
        if centrality_type not in ["degree", "betweenness", "closeness", "eigenvector"]:
            msg = f"Invalid centrality type: {centrality_type}"
            raise ValueError(msg)

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
            # Use degree centrality which returns a dictionary
            # Convert to dict[str, float] to satisfy type checking
            return {str(node): float(value) for node, value in nx.degree_centrality(graph).items()}
        if centrality_type == "betweenness":
            return {
                str(node): float(value) for node, value in nx.betweenness_centrality(graph).items()
            }
        if centrality_type == "closeness":
            return {
                str(node): float(value) for node, value in nx.closeness_centrality(graph).items()
            }
        if centrality_type == "eigenvector":
            return {
                str(node): float(value)
                for node, value in nx.eigenvector_centrality(graph, max_iter=1000).items()
            }

        # This should never happen due to the validation at the beginning
        return {}

    def save(self, directory: str) -> None:
        """
        Save the graph to disk.

        Args:
        ----
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
        ----
            directory: Directory to load from

        Raises:
        ------
            FileNotFoundError: If the graph files are not found

        """
        directory_path = Path(directory)

        # Check if files exist
        nodes_path = directory_path / "nodes.json"
        edges_path = directory_path / "edges.json"
        graph_path = directory_path / "graph.gpickle"

        if not nodes_path.exists() or not edges_path.exists() or not graph_path.exists():
            msg = f"Graph files not found in {directory}"
            raise FileNotFoundError(msg)

        # Clear existing data
        self.nodes.clear()
        self.edges.clear()
        self.graph = nx.DiGraph()

        # Load graph structure
        import pickle

        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)

        # Load nodes
        with open(nodes_path) as f:
            nodes_data = json.load(f)

        for node_id, node_data in nodes_data.items():
            node_type = node_data["type"]

            if node_type == "DocumentNode":
                # We need to load the document separately
                node_data["document_id"]
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
        with open(edges_path) as f:
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the graph to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation

        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {
                f"{source_id}|{target_id}|{rel_type}": edge.to_dict()
                for (source_id, target_id, rel_type), edge in self.edges.items()
            },
        }

    def __len__(self):
        """
        Get the number of nodes in the graph.

        Returns
        -------
            int: Number of nodes

        """
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        """
        Check if a node is in the graph.

        Args:
        ----
            node_id: ID of the node

        Returns:
        -------
            bool: True if the node is in the graph, False otherwise

        """
        return node_id in self.nodes

    def build_from_memory(self, memory_store: Any) -> None:
        """
        Build the dependency graph from a memory store.

        This method analyzes the documents in the memory store and creates
        relationships between them based on their content and metadata.

        Args:
        ----
            memory_store: Memory store containing the documents

        """
        import asyncio

        logger.info("Building dependency graph from memory store")

        # Get all documents from memory store
        documents = asyncio.run(memory_store.get_all_documents())

        if not documents:
            logger.warning("No documents found in memory store")
            return

        logger.info(f"Found {len(documents)} documents in memory store")

        # Add all documents to the graph
        for document in documents:
            if document.id not in self.nodes:
                self.add_document_node(document)

        # Create relationships between documents based on their content and metadata
        self._build_code_relationships(documents)

        logger.info(
            f"Built dependency graph with {len(self.nodes)} nodes and {len(self.edges)} edges"
        )

    def _build_code_relationships(self, documents: list["Document"]) -> None:
        """
        Build relationships between code documents.

        Args:
        ----
            documents: List of documents to analyze

        """
        # Create a mapping of file paths to document IDs
        path_to_id = {}
        for doc in documents:
            if doc.metadata:
                # Get source from metadata
                source = _get_source_from_metadata(doc.metadata)

                if source:
                    path_to_id[source] = doc.id

        # Create a mapping of module names to document IDs
        module_to_id = {}
        for doc in documents:
            if doc.metadata:
                # Get source from metadata
                source = _get_source_from_metadata(doc.metadata)

                if source and source.endswith(".py"):
                    # For Python files, use the file name without extension as module name
                    module_name = source.split("/")[-1].replace(".py", "")
                    module_to_id[module_name] = doc.id

                    # Also handle package imports
                    parts = source.split("/")
                    if len(parts) > 1:
                        # Create package path
                        for i in range(1, len(parts)):
                            package = ".".join(parts[:-i])
                            module_to_id[package] = doc.id

        # Analyze each document for imports and dependencies
        for doc in documents:
            if not doc.metadata:
                continue

            # Get source from metadata
            source = _get_source_from_metadata(doc.metadata)

            if not source:
                continue

            doc_id = doc.id

            # Handle Python files
            if source.endswith(".py"):
                self._analyze_python_imports(doc, doc_id, module_to_id)

            # Handle JavaScript/TypeScript files
            elif source.endswith((".js", ".ts")):
                self._analyze_js_imports(doc, doc_id, path_to_id)

            # Handle Java files
            elif source.endswith(".java"):
                self._analyze_java_imports(doc, doc_id, path_to_id)

            # Handle C/C++ files
            elif source.endswith((".c", ".cpp", ".h", ".hpp")):
                self._analyze_cpp_includes(doc, doc_id, path_to_id)

    def _analyze_python_imports(
        self, doc: "Document", doc_id: str, module_to_id: dict[str, str]
    ) -> None:
        """
        Analyze Python imports and create relationships.

        Args:
        ----
            doc: Document to analyze
            doc_id: ID of the document
            module_to_id: Mapping of module names to document IDs

        """
        import re

        # Regular expressions for imports
        import_pattern = r"^\s*import\s+([\w\.]+)(?:\s+as\s+\w+)?\s*$"
        from_import_pattern = r"^\s*from\s+([\w\.]+)\s+import\s+"

        # Find all imports
        for line in doc.content.split("\n"):
            # Check for 'import' statements
            import_match = re.match(import_pattern, line)
            if import_match:
                module_name = import_match.group(1)
                self._add_import_relationship(doc_id, module_name, module_to_id)
                continue

            # Check for 'from ... import' statements
            from_match = re.match(from_import_pattern, line)
            if from_match:
                module_name = from_match.group(1)
                self._add_import_relationship(doc_id, module_name, module_to_id)

    def _add_import_relationship(
        self, source_id: str, module_name: str, module_to_id: dict[str, str]
    ) -> None:
        """
        Add an import relationship between documents.

        Args:
        ----
            source_id: ID of the source document
            module_name: Name of the imported module
            module_to_id: Mapping of module names to document IDs

        """
        # Check if the module exists in our mapping
        if module_name in module_to_id:
            target_id = module_to_id[module_name]

            # Don't create self-references
            if source_id != target_id:
                try:
                    # Create a relationship object
                    relationship = Relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type="imports",
                        metadata={"module": module_name, "confidence": 1.0},
                    )
                    self.add_relationship(relationship)
                except ValueError as e:
                    logger.debug(f"Failed to add import relationship: {e}")

    def _analyze_js_imports(self, doc: "Document", doc_id: str, path_to_id: dict[str, str]) -> None:
        """
        Analyze JavaScript/TypeScript imports and create relationships.

        Args:
        ----
            doc: Document to analyze
            doc_id: ID of the document
            path_to_id: Mapping of file paths to document IDs

        """
        import re

        # Regular expressions for imports
        import_patterns = [
            r"import\s+.*\s+from\s+['\"](.+?)['\"]",  # ES6 imports
            r"require\s*\(\s*['\"](.+?)['\"]",  # CommonJS require
        ]

        for line in doc.content.split("\n"):
            for pattern in import_patterns:
                for match in re.finditer(pattern, line):
                    path = match.group(1)

                    # Handle relative imports
                    if path.startswith("."):
                        # Get source from metadata
                        source_path = _get_source_from_metadata(doc.metadata)

                        if source_path:
                            source_dir = "/".join(source_path.split("/")[:-1])

                            if path.startswith("./"):
                                path = f"{source_dir}/{path[2:]}"
                            elif path.startswith("../"):
                                parts = path.split("/")
                                up_count = 0
                                for part in parts:
                                    if part == "..":
                                        up_count += 1
                                    else:
                                        break

                                source_parts = source_dir.split("/")
                                if up_count < len(source_parts):
                                    base_path = "/".join(source_parts[:-up_count])
                                    rel_path = "/".join(parts[up_count:])
                                    path = f"{base_path}/{rel_path}"

                    # Add extensions if missing
                    if not path.endswith((".js", ".ts")):
                        js_path = f"{path}.js"
                        ts_path = f"{path}.ts"

                        if js_path in path_to_id:
                            path = js_path
                        elif ts_path in path_to_id:
                            path = ts_path

                    # Add relationship if the target exists
                    if path in path_to_id:
                        target_id = path_to_id[path]
                        if doc_id != target_id:
                            try:
                                # Create a relationship object
                                relationship = Relationship(
                                    source_id=doc_id,
                                    target_id=target_id,
                                    relationship_type="imports",
                                    metadata={"module": path, "confidence": 1.0},
                                )
                                self.add_relationship(relationship)
                            except ValueError as e:
                                logger.debug(f"Failed to add import relationship: {e}")

    def _analyze_java_imports(
        self, doc: "Document", doc_id: str, path_to_id: dict[str, str]
    ) -> None:
        """
        Analyze Java imports and create relationships.

        Args:
        ----
            doc: Document to analyze
            doc_id: ID of the document
            path_to_id: Mapping of file paths to document IDs

        """
        import re

        # Regular expression for imports
        import_pattern = r"import\s+([\w\.]+)(?:\.\*)?;"

        # Create a mapping of package names to document IDs
        package_to_id = {}
        for path, doc_id in path_to_id.items():
            if path.endswith(".java"):
                # Extract package from file content
                package_match = re.search(r"package\s+([\w\.]+);", doc.content)
                if package_match:
                    package = package_match.group(1)
                    package_to_id[package] = doc_id

        # Find all imports
        for line in doc.content.split("\n"):
            import_match = re.search(import_pattern, line)
            if import_match:
                package = import_match.group(1)

                # Check if the package exists in our mapping
                if package in package_to_id:
                    target_id = package_to_id[package]
                    if doc_id != target_id:
                        try:
                            # Create a relationship object
                            relationship = Relationship(
                                source_id=doc_id,
                                target_id=target_id,
                                relationship_type="imports",
                                metadata={"package": package, "confidence": 1.0},
                            )
                            self.add_relationship(relationship)
                        except ValueError as e:
                            logger.debug(f"Failed to add import relationship: {e}")

    def _analyze_cpp_includes(
        self, doc: "Document", doc_id: str, path_to_id: dict[str, str]
    ) -> None:
        """
        Analyze C/C++ includes and create relationships.

        Args:
        ----
            doc: Document to analyze
            doc_id: ID of the document
            path_to_id: Mapping of file paths to document IDs

        """
        import re

        # Regular expression for includes
        include_pattern = r'#include\s+[<"](.+?)[>"]'

        # Create a mapping of header file names to document IDs
        header_to_id = {}
        for path, doc_id in path_to_id.items():
            if path.endswith((".h", ".hpp")):
                header_name = path.split("/")[-1]
                header_to_id[header_name] = doc_id

        # Find all includes
        for line in doc.content.split("\n"):
            include_match = re.search(include_pattern, line)
            if include_match:
                header = include_match.group(1)

                # Check if the header exists in our mapping
                if header in header_to_id:
                    target_id = header_to_id[header]
                    if doc_id != target_id:
                        try:
                            # Create a relationship object
                            relationship = Relationship(
                                source_id=doc_id,
                                target_id=target_id,
                                relationship_type="includes",
                                metadata={"header": header, "confidence": 1.0},
                            )
                            self.add_relationship(relationship)
                        except ValueError as e:
                            logger.debug(f"Failed to add include relationship: {e}")
