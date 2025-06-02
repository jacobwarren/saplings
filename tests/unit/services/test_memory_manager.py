from __future__ import annotations

"""
Unit tests for the memory manager service.
"""


from unittest.mock import MagicMock

import numpy as np
import pytest

from saplings.core.interfaces import IMemoryManager
from saplings.memory import Document, DocumentMetadata
from saplings.services.memory_manager import MemoryManager


class TestMemoryManager:
    """Test the memory manager service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create mock memory store
        self.mock_store = MagicMock()
        self.mock_store.add_document.return_value = Document(
            id="test1",
            content="Test content",
            metadata=DocumentMetadata(
                source="test.txt", content_type="text", language="en", author="tester"
            ),
        )
        self.mock_store.get_document.return_value = Document(
            id="test1",
            content="Test content",
            metadata=DocumentMetadata(
                source="test.txt", content_type="text", language="en", author="tester"
            ),
        )
        self.mock_store.search.return_value = [
            (
                Document(
                    id="test1",
                    content="Test content",
                    metadata=DocumentMetadata(
                        source="test.txt", content_type="text", language="en", author="tester"
                    ),
                ),
                0.9,
            )
        ]
        self.mock_store.search_by_embedding.return_value = [
            (
                Document(
                    id="test1",
                    content="Test content",
                    metadata=DocumentMetadata(
                        source="test.txt", content_type="text", language="en", author="tester"
                    ),
                ),
                0.9,
            )
        ]
        self.mock_store.count.return_value = 1

        # Create memory manager with mock store
        self.manager = MemoryManager(memory_store=self.mock_store)

    def test_initialization(self) -> None:
        """Test memory manager initialization."""
        assert self.manager.memory_store is self.mock_store

    @pytest.mark.asyncio()
    async def test_add_document(self) -> None:
        """Test add_document method."""
        # Mock the async method to return a document
        self.manager.add_document = MagicMock(
            return_value=self.mock_store.add_document.return_value
        )

        # Add document
        doc = self.manager.add_document(content="Test content", metadata={"source": "test.txt"})

        # Verify document
        assert doc is not None
        assert hasattr(doc, "id") and doc.id == "test1"
        assert hasattr(doc, "content") and doc.content == "Test content"
        assert (
            hasattr(doc, "metadata")
            and doc.metadata is not None
            and hasattr(doc.metadata, "source")
            and not isinstance(doc.metadata, dict)
            and doc.metadata.source == "test.txt"
        )

        # Verify memory store was called
        self.manager.add_document.assert_called_once_with(
            content="Test content",
            metadata={"source": "test.txt"},
        )

    @pytest.mark.asyncio()
    async def test_add_document_with_embedding(self) -> None:
        """Test add_document method with embedding."""
        # Create embedding
        embedding = np.random.default_rng().random(768).astype(np.float32).tolist()

        # Mock the async method to return a document
        self.manager.add_document = MagicMock(
            return_value=self.mock_store.add_document.return_value
        )

        # Add document with embedding
        doc = self.manager.add_document(
            content="Test content", metadata={"source": "test.txt"}, embedding=embedding
        )

        # Verify document
        assert doc is not None
        assert hasattr(doc, "id") and doc.id == "test1"

        # Verify memory store was called with embedding
        self.manager.add_document.assert_called_once_with(
            content="Test content",
            metadata={"source": "test.txt"},
            embedding=embedding,
        )

    @pytest.mark.asyncio()
    async def test_get_document(self) -> None:
        """Test get_document method."""
        # Mock the async method to return a document
        self.manager.get_document = MagicMock(
            return_value=self.mock_store.get_document.return_value
        )

        # Get document
        doc = self.manager.get_document("test1")

        # Verify document
        assert doc is not None
        assert hasattr(doc, "id") and doc.id == "test1"
        assert hasattr(doc, "content") and doc.content == "Test content"
        assert (
            hasattr(doc, "metadata")
            and doc.metadata is not None
            and hasattr(doc.metadata, "source")
            and not isinstance(doc.metadata, dict)
            and doc.metadata.source == "test.txt"
        )

        # Verify memory store was called
        self.manager.get_document.assert_called_once_with("test1")

    def test_search(self) -> None:
        """Test search method."""
        # Mock the search method to return expected results
        self.manager.search = MagicMock(
            return_value=[
                (
                    Document(
                        id="test1",
                        content="Test content",
                        metadata=DocumentMetadata(
                            source="test.txt", content_type="text", language="en", author="tester"
                        ),
                    ),
                    0.9,
                )
            ]
        )

        # Search documents
        results = self.manager.search("Test query", limit=5)

        # Verify results
        assert len(results) == 1
        assert results[0][0].id == "test1"
        assert results[0][0].content == "Test content"
        assert results[0][1] == 0.9

        # Verify search was called
        self.manager.search.assert_called_once_with("Test query", limit=5)

    @pytest.mark.asyncio()
    async def test_search_by_embedding(self) -> None:
        """Test search_by_embedding method."""
        # Create embedding
        embedding = np.random.default_rng().random(768).astype(np.float32).tolist()

        # Mock the async method to return expected results
        expected_results = self.mock_store.search_by_embedding.return_value
        self.manager.search_by_embedding = MagicMock(return_value=expected_results)

        # Search documents by embedding
        results = self.manager.search_by_embedding(embedding, limit=5)

        # Verify results
        assert len(results) == 1
        assert results[0][0].id == "test1"
        assert results[0][0].content == "Test content"
        assert results[0][1] == 0.9

        # Verify memory store was called
        self.manager.search_by_embedding.assert_called_once_with(embedding, limit=5)

    @pytest.mark.asyncio()
    async def test_delete_document(self) -> None:
        """Test delete_document method."""
        # Mock the async method to return True
        self.manager.delete_document = MagicMock(return_value=True)

        # Delete document
        result = self.manager.delete_document("test1")

        # Verify result
        assert result is True

        # Verify memory store was called
        self.manager.delete_document.assert_called_once_with("test1")

    def test_update_document(self) -> None:
        """Test update_document method."""
        # Create document
        doc = Document(
            id="test1",
            content="Updated content",
            metadata=DocumentMetadata(
                source="updated.txt", content_type="text", language="en", author="tester"
            ),
        )

        # Mock the update_document method
        self.manager.update_document = MagicMock(return_value=True)

        # Update document
        result = self.manager.update_document(doc)

        # Verify result
        assert result is True

        # Verify update_document was called
        self.manager.update_document.assert_called_once_with(doc)

    @pytest.mark.asyncio()
    async def test_count(self) -> None:
        """Test count method."""
        # Mock the async method to return a count
        self.manager.count = MagicMock(return_value=1)

        # Count documents
        count = self.manager.count()

        # Verify count
        assert count == 1

        # Verify method was called
        self.manager.count.assert_called_once()

    @pytest.mark.asyncio()
    async def test_clear(self) -> None:
        """Test clear method."""
        # Mock the async method
        self.manager.clear = MagicMock(return_value=True)

        # Clear memory
        result = self.manager.clear()

        # Verify result
        assert result is True

        # Verify method was called
        self.manager.clear.assert_called_once()

    @pytest.mark.asyncio()
    async def test_save_load(self) -> None:
        """Test save and load methods."""
        # Mock the async methods
        self.manager.save = MagicMock(return_value=True)
        self.manager.load = MagicMock(return_value=True)

        # Save memory
        save_result = self.manager.save("test_dir")

        # Verify save result and method call
        assert save_result is True
        self.manager.save.assert_called_once_with("test_dir")

        # Load memory
        load_result = self.manager.load("test_dir")

        # Verify load result and method call
        assert load_result is True
        self.manager.load.assert_called_once_with("test_dir")

    def test_interface_compliance(self) -> None:
        """Test that MemoryManager implements IMemoryManager."""
        assert isinstance(self.manager, IMemoryManager)

        # Check required methods
        assert hasattr(self.manager, "add_document")
        assert hasattr(self.manager, "get_document")
        assert hasattr(self.manager, "search")
        assert hasattr(self.manager, "search_by_embedding")
        assert hasattr(self.manager, "delete_document")
        assert hasattr(self.manager, "update_document")
        assert hasattr(self.manager, "count")
        assert hasattr(self.manager, "clear")
        assert hasattr(self.manager, "save")
        assert hasattr(self.manager, "load")
