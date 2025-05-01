"""
Tests for the graph expander module.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from saplings.memory.document import Document, DocumentMetadata
from saplings.memory.graph import DependencyGraph, DocumentNode, EntityNode
from saplings.memory.indexer import Entity
from saplings.memory.memory_store import MemoryStore
from saplings.retrieval.config import GraphConfig, RetrievalConfig
from saplings.retrieval.graph_expander import GraphExpander


class TestGraphExpander:
    """Tests for the GraphExpander class."""

    def setup_method(self):
        """Set up test environment."""
        # Create a mock memory store
        self.memory_store = MagicMock(spec=MemoryStore)

        # Create test documents
        self.docs = [
            Document(
                id="doc1",
                content="This is document 1.",
                metadata=DocumentMetadata(source="test1.txt"),
            ),
            Document(
                id="doc2",
                content="This is document 2.",
                metadata=DocumentMetadata(source="test2.txt"),
            ),
            Document(
                id="doc3",
                content="This is document 3.",
                metadata=DocumentMetadata(source="test3.txt"),
            ),
            Document(
                id="doc4",
                content="This is document 4.",
                metadata=DocumentMetadata(source="test4.txt"),
            ),
        ]

        # Create a mock graph
        self.graph = MagicMock(spec=DependencyGraph)

        # Configure the mock memory store
        self.memory_store.graph = self.graph

        # Create the graph expander
        self.config = GraphConfig(
            max_hops=2,
            max_nodes=10,
            min_edge_weight=0.5,
            score_decay_factor=0.8,
        )
        self.expander = GraphExpander(self.memory_store, self.config)

    def test_init_with_retrieval_config(self):
        """Test initialization with RetrievalConfig."""
        retrieval_config = RetrievalConfig(graph=GraphConfig(max_hops=3))
        expander = GraphExpander(self.memory_store, retrieval_config)

        assert expander.config.max_hops == 3

    def test_expand_empty_documents(self):
        """Test expanding an empty document list."""
        results = self.expander.expand([])

        assert results == []

    def test_expand_with_subgraph(self):
        """Test expanding documents using a subgraph."""
        # Create a mock subgraph
        subgraph = MagicMock(spec=DependencyGraph)

        # Configure the mock graph
        self.graph.get_subgraph.return_value = subgraph

        # Create mock nodes
        doc1_node = MagicMock(spec=DocumentNode)
        doc1_node.id = "doc1"
        doc1_node.document = self.docs[0]

        doc3_node = MagicMock(spec=DocumentNode)
        doc3_node.id = "doc3"
        doc3_node.document = self.docs[2]

        entity_node = MagicMock(spec=EntityNode)
        entity_node.id = "entity:person:John"

        # Configure the mock subgraph
        subgraph.nodes = {
            "doc1": doc1_node,
            "doc3": doc3_node,
            "entity:person:John": entity_node,
        }

        # Configure find_paths to return a path
        subgraph.find_paths.return_value = [
            [
                ("doc1", "mentions", "entity:person:John"),
                ("entity:person:John", "mentioned_in", "doc3"),
            ]
        ]

        # Configure the graph data
        subgraph.graph = nx.DiGraph()
        subgraph.graph.add_edge(
            "doc1", "entity:person:John", relationship_type="mentions", weight=0.9
        )
        subgraph.graph.add_edge(
            "entity:person:John", "doc3", relationship_type="mentioned_in", weight=0.8
        )

        # Expand documents
        results = self.expander.expand([self.docs[0]], [1.0])

        # Check that the graph was queried
        self.graph.get_subgraph.assert_called_once_with(
            node_ids=["doc1"],
            max_hops=2,
            relationship_types=None,
        )

        # Check results
        assert len(results) == 2
        assert results[0][0].id == "doc1"  # Original document
        assert results[1][0].id == "doc3"  # Expanded document
        assert results[0][1] == 1.0  # Original score
        assert results[1][1] < 1.0  # Expanded score should be lower

    def test_calculate_score(self):
        """Test calculating a score for a node."""
        # Create a mock subgraph
        subgraph = MagicMock(spec=DependencyGraph)

        # Configure find_paths to return a path
        subgraph.find_paths.return_value = [
            [
                ("doc1", "mentions", "entity:person:John"),
                ("entity:person:John", "mentioned_in", "doc3"),
            ]
        ]

        # Configure the graph data
        subgraph.graph = nx.DiGraph()
        subgraph.graph.add_edge(
            "doc1", "entity:person:John", relationship_type="mentions", weight=0.9
        )
        subgraph.graph.add_edge(
            "entity:person:John", "doc3", relationship_type="mentioned_in", weight=0.8
        )

        # Calculate score
        score = self.expander._calculate_score(
            node_id="doc3",
            seed_doc_ids=["doc1"],
            doc_scores={"doc1": 1.0},
            subgraph=subgraph,
        )

        # Check score
        assert score > 0.0
        assert score < 1.0  # Should be decayed

    def test_save_and_load(self):
        """Test saving and loading the graph expander."""
        # Save the expander
        with tempfile.TemporaryDirectory() as temp_dir:
            self.expander.save(temp_dir)

            # Create a new expander and load the saved data
            new_expander = GraphExpander(self.memory_store)
            new_expander.load(temp_dir)

            # Check that the loaded expander has the same configuration
            assert new_expander.config.max_hops == self.expander.config.max_hops
            assert new_expander.config.max_nodes == self.expander.config.max_nodes
            assert new_expander.config.min_edge_weight == self.expander.config.min_edge_weight
            assert new_expander.config.score_decay_factor == self.expander.config.score_decay_factor
