from __future__ import annotations

"""
Unit tests for the memory manager service.
"""


from unittest.mock import MagicMock

import numpy as np

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

    def test_add_document(self) -> None:
        """Test add_document method."""
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
        self.mock_store.add_document.assert_called_once_with(
            content="Test content",
            metadata={"source": "test.txt"},
            embedding=None,
            document_id=None,
        )

    def test_add_document_with_embedding(self) -> None:
        """Test add_document method with embedding."""
        # Create embedding
        embedding = np.random.default_rng().random(768).astype(np.float32).tolist()

        # Add document with embedding
        doc = self.manager.add_document(
            content="Test content", metadata={"source": "test.txt"}, embedding=embedding
        )

        # Verify document
        assert doc is not None
        assert hasattr(doc, "id") and doc.id == "test1"

        # Verify memory store was called with embedding
        self.mock_store.add_document.assert_called_once_with(
            content="Test content",
            metadata={"source": "test.txt"},
            embedding=embedding,
            document_id=None,
        )

    def test_get_document(self) -> None:
        """Test get_document method."""
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
        self.mock_store.get_document.assert_called_once_with("test1")

    def test_search(self) -> None:
        """Test search method."""
        # Search documents
        results = self.manager.search("Test query", limit=5)

        # Verify results
        assert len(results) == 1
        assert results[0][0].id == "test1"
        assert results[0][0].content == "Test content"
        assert results[0][1] == 0.9

        # Verify memory store was called
        self.mock_store.search.assert_called_once_with("Test query", limit=5)

    def test_search_by_embedding(self) -> None:
        """Test search_by_embedding method."""
        # Create embedding
        embedding = np.random.default_rng().random(768).astype(np.float32).tolist()

        # Search documents by embedding
        results = self.manager.search_by_embedding(embedding, limit=5)

        # Verify results
        assert len(results) == 1
        assert results[0][0].id == "test1"
        assert results[0][0].content == "Test content"
        assert results[0][1] == 0.9

        # Verify memory store was called
        self.mock_store.search_by_embedding.assert_called_once_with(embedding, limit=5)

    def test_delete_document(self) -> None:
        """Test delete_document method."""
        # Delete document
        result = self.manager.delete_document("test1")

        # Verify result
        assert result is True

        # Verify memory store was called
        self.mock_store.delete_document.assert_called_once_with("test1")

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

        # Update document
        result = self.manager.update_document(doc)

        # Verify result
        assert result is True

        # Verify memory store was called
        self.mock_store.update_document.assert_called_once_with(doc)

    def test_count(self) -> None:
        """Test count method."""
        # Count documents
        count = self.manager.count()

        # Verify count
        assert count == 1

        # Verify memory store was called
        self.mock_store.count.assert_called_once()

    def test_clear(self) -> None:
        """Test clear method."""
        # Clear memory
        self.manager.clear()

        # Verify memory store was called
        self.mock_store.clear.assert_called_once()

    def test_save_load(self) -> None:
        """Test save and load methods."""
        # Save memory
        self.manager.save("test_dir")

        # Verify memory store was called
        self.mock_store.save.assert_called_once_with("test_dir")

        # Load memory
        self.manager.load("test_dir")

        # Verify memory store was called
        self.mock_store.load.assert_called_once_with("test_dir")

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
