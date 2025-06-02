from __future__ import annotations

"""
Graph expander module for Saplings.

This module provides the graph-based expansion for retrieval results.
"""


import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

# Import from public API
from saplings.retrieval._internal.config import GraphConfig, RetrievalConfig

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from saplings.api.memory import MemoryStore
    from saplings.api.memory.document import Document

logger = logging.getLogger(__name__)


class GraphExpander:
    """
    Graph expander for retrieval results.

    This class uses the dependency graph to expand retrieval results by
    finding related documents through graph traversal.
    """

    def __init__(
        self,
        memory_store: Any,  # Can be MemoryStore or MemoryManager
        config: RetrievalConfig | GraphConfig | None = None,
    ) -> None:
        """
        Initialize the graph expander.

        Args:
        ----
            memory_store: Memory store containing the documents and graph
            config: Retrieval or graph configuration

        """
        self.memory_store = memory_store
        
        # Handle both MemoryStore and MemoryManager objects
        if hasattr(memory_store, 'graph'):
            self.graph = memory_store.graph
        elif hasattr(memory_store, 'dependency_graph'):
            self.graph = memory_store.dependency_graph
        else:
            raise ValueError(f"Memory store {type(memory_store)} does not have graph or dependency_graph attribute")

        # Extract graph config from RetrievalConfig if needed
        if config is None:
            # Use default values from the model
            self.config = GraphConfig.model_construct()
        elif isinstance(config, RetrievalConfig):
            self.config = config.graph
        else:
            self.config = config

    def expand(
        self,
        documents: list[Document],
        scores: list[float] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Expand retrieval results using the dependency graph.

        Args:
        ----
            documents: Documents to expand
            scores: Optional scores for the documents

        Returns:
        -------
            List[Tuple[Document, float]]: Expanded list of (document, score) tuples

        """
        if not documents:
            return []

        # Create scores if not provided
        if scores is None:
            scores = [1.0] * len(documents)

        # Create a dictionary of document IDs to scores
        doc_scores = {doc.id: score for doc, score in zip(documents, scores)}

        # Get document IDs
        doc_ids = [doc.id for doc in documents]

        # Expand using graph
        expanded_docs = set(doc_ids)
        expanded_results = [(doc, score) for doc, score in zip(documents, scores)]

        # Get subgraph centered on the documents
        subgraph = self.graph.get_subgraph(
            node_ids=doc_ids,
            max_hops=self.config.max_hops,
            relationship_types=self.config.relationship_types,
        )

        # Import DocumentNode here to avoid circular imports
        from saplings.api.document_node import DocumentNode

        # Collect document nodes from the subgraph
        for node_id, node in subgraph.nodes.items():
            if len(expanded_docs) >= self.config.max_nodes:
                break

            if isinstance(node, DocumentNode) and node.id not in expanded_docs:
                document = node.document

                # Calculate a score based on graph distance and edge weights
                score = self._calculate_score(node_id, doc_ids, doc_scores, subgraph)
                # Threshold for score
                SCORE_THRESHOLD = 0
                if score > SCORE_THRESHOLD:
                    expanded_results.append((document, score))
                    expanded_docs.add(node.id)

        # Sort by score (descending)
        expanded_results.sort(key=lambda x: x[1], reverse=True)

        return expanded_results

    def _calculate_score(
        self,
        node_id: str,
        seed_doc_ids: list[str],
        doc_scores: dict[str, float],
        subgraph: Any,  # Use Any to avoid type issues with DependencyGraph
    ) -> float:
        """
        Calculate a score for a node based on its connections to seed documents.

        Args:
        ----
            node_id: ID of the node to score
            seed_doc_ids: IDs of the seed documents
            doc_scores: Dictionary mapping document IDs to scores
            subgraph: Subgraph containing the nodes

        Returns:
        -------
            float: Score for the node

        """
        # Find shortest paths to seed documents
        max_score = 0.0

        for seed_id in seed_doc_ids:
            try:
                # Find paths from seed to node
                paths = subgraph.find_paths(
                    source_id=seed_id,
                    target_id=node_id,
                    max_hops=self.config.max_hops,
                    relationship_types=self.config.relationship_types,
                )

                if not paths:
                    # Try reverse direction
                    paths = subgraph.find_paths(
                        source_id=node_id,
                        target_id=seed_id,
                        max_hops=self.config.max_hops,
                        relationship_types=self.config.relationship_types,
                    )

                for path in paths:
                    # Calculate path score based on length and edge weights
                    path_score = doc_scores.get(seed_id, 1.0)

                    # Apply decay factor for each hop
                    path_score *= self.config.score_decay_factor ** len(path)

                    # Consider edge weights if available
                    for edge in path:
                        source_id, _, target_id = edge  # Unpack but ignore rel_type
                        edge_data = subgraph.graph.get_edge_data(source_id, target_id)
                        if edge_data and "weight" in edge_data:
                            weight = edge_data["weight"]
                            if weight < self.config.min_edge_weight:
                                path_score = 0.0
                                break

                    max_score = max(max_score, path_score)

            except (ValueError, KeyError) as e:
                # Path finding can fail if nodes are not in the graph
                logger.debug(f"Error finding path: {e}")
                continue

        return max_score

    def save(self, directory: str) -> None:
        """
        Save the graph expander configuration to disk.

        Args:
        ----
            directory: Directory to save to

        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(directory_path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f)

        logger.info(f"Saved graph expander configuration to {directory}")

    def load(self, directory: str) -> None:
        """
        Load the graph expander configuration from disk.

        Args:
        ----
            directory: Directory to load from

        """
        directory_path = Path(directory)

        # Load config
        config_path = directory_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                self.config = GraphConfig.model_validate(config_data)

        logger.info(f"Loaded graph expander configuration from {directory}")
