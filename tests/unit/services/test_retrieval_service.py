from __future__ import annotations

"""
Unit tests for the retrieval service.
"""


from unittest.mock import MagicMock

import numpy as np

from saplings.core.interfaces import IMemoryManager, IRetrievalService
from saplings.memory import Document
from saplings.retrieval.config import RetrievalConfig
from saplings.services.retrieval_service import RetrievalService


class TestRetrievalService:
    EXPECTED_COUNT_1 = 2

    """Test the retrieval service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create mock memory manager
        self.mock_memory = MagicMock(spec=IMemoryManager)
        self.mock_memory.search.return_value = [
            (Document(id="test1", content="Test content 1"), 0.9),
            (Document(id="test2", content="Test content 2"), 0.8),
        ]
        self.mock_memory.search_by_embedding.return_value = [
            (Document(id="test1", content="Test content 1"), 0.9),
            (Document(id="test2", content="Test content 2"), 0.8),
        ]

        # Create mock embedding function
        self.mock_get_embedding = MagicMock(return_value=np.random.rand(768).astype(np.float32))

        # Create retrieval service
        self.config = RetrievalConfig(
            retrieval_type="hybrid", top_k=5, similarity_threshold=0.7, reranking_enabled=False
        )
        self.service = RetrievalService(
            memory_manager=self.mock_memory,
            config=self.config,
            get_embedding=self.mock_get_embedding,
        )

    def test_initialization(self) -> None:
        """Test retrieval service initialization."""
        assert self.service.memory_manager is self.mock_memory
        assert self.service.config is self.config
        assert self.service.get_embedding is self.mock_get_embedding

    def test_retrieve(self) -> None:
        """Test retrieve method."""
        # Retrieve documents
        results = self.service.retrieve("Test query", limit=5)

        # Verify results
        assert len(results) == self.EXPECTED_COUNT_1
        assert results[0][0].id == "test1"
        assert results[0][0].content == "Test content 1"
        assert results[0][1] == 0.9
        assert results[1][0].id == "test2"
        assert results[1][0].content == "Test content 2"
        assert results[1][1] == 0.8

        # Verify memory manager was called
        self.mock_memory.search.assert_called_once_with("Test query", limit=5)

    def test_retrieve_with_embedding(self) -> None:
        """Test retrieve_with_embedding method."""
        # Create embedding
        embedding = np.random.rand(768).astype(np.float32)

        # Retrieve documents with embedding
        results = self.service.retrieve_with_embedding(embedding, limit=5)

        # Verify results
        assert len(results) == self.EXPECTED_COUNT_1
        assert results[0][0].id == "test1"
        assert results[0][0].content == "Test content 1"
        assert results[0][1] == 0.9
        assert results[1][0].id == "test2"
        assert results[1][0].content == "Test content 2"
        assert results[1][1] == 0.8

        # Verify memory manager was called
        self.mock_memory.search_by_embedding.assert_called_once_with(embedding, limit=5)

    def test_retrieve_with_filter(self) -> None:
        """Test retrieve method with filter."""
        # Retrieve documents with filter
        results = self.service.retrieve("Test query", limit=5, filter_dict={"source": "test.txt"})

        # Verify results
        assert len(results) == self.EXPECTED_COUNT_1

        # Verify memory manager was called with filter
        self.mock_memory.search.assert_called_once_with(
            "Test query", limit=5, filter_dict={"source": "test.txt"}
        )

    def test_retrieve_with_threshold(self) -> None:
        """Test retrieve method with threshold."""
        # Set up mock memory manager to return results with scores below threshold
        self.mock_memory.search.return_value = [
            (Document(id="test1", content="Test content 1"), 0.9),
            (Document(id="test2", content="Test content 2"), 0.6),  # Below threshold
        ]

        # Retrieve documents with threshold
        results = self.service.retrieve("Test query", limit=5)

        # Verify results (only documents above threshold)
        assert len(results) == 1
        assert results[0][0].id == "test1"
        assert results[0][1] == 0.9

    def test_interface_compliance(self) -> None:
        """Test that RetrievalService implements IRetrievalService."""
        assert isinstance(self.service, IRetrievalService)

        # Check required methods
        assert hasattr(self.service, "retrieve")
        assert hasattr(self.service, "retrieve_with_embedding")
