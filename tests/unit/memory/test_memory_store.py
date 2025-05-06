from __future__ import annotations

import tempfile
from unittest.mock import patch

import numpy as np

from saplings.memory import Document, DocumentMetadata, MemoryStore
from saplings.memory.config import MemoryConfig

"""
Unit tests for the memory store.
"""


class TestMemoryStore:
    EXPECTED_COUNT_1 = 3

    """Test the memory store."""

    def test_memory_store_initialization(self) -> None:
        """Test memory store initialization."""
        # Test with default configuration
        memory = MemoryStore()
        assert memory.config is not None
        assert memory.vector_store is not None
        assert memory.graph is not None
        assert memory.indexer is not None

        # Test with custom configuration
        config = MemoryConfig(chunk_size=500, chunk_overlap=100)
        memory = MemoryStore(config=config)
        assert memory.config.chunk_size == 500
        assert memory.config.chunk_overlap == 100

    def test_add_document(self) -> None:
        """Test adding documents to memory store."""
        memory = MemoryStore()

        # Add a document
        doc = memory.add_document(content="Test document content", metadata={"source": "test.txt"})

        # Verify document properties
        assert doc is not None
        assert hasattr(doc, "id")
        assert doc.content == "Test document content"
        if (
            hasattr(doc, "metadata")
            and hasattr(doc.metadata, "source")
            and not isinstance(doc.metadata, dict)
        ):
            assert doc.metadata.source == "test.txt"

        # Test retrieval
        retrieved_doc = memory.get_document(doc.id)
        assert retrieved_doc is not None
        assert hasattr(retrieved_doc, "id") and retrieved_doc.id == doc.id
        assert hasattr(retrieved_doc, "content") and retrieved_doc.content == doc.content
        if (
            hasattr(retrieved_doc, "metadata")
            and hasattr(retrieved_doc.metadata, "source")
            and not isinstance(retrieved_doc.metadata, dict)
        ):
            assert retrieved_doc.metadata.source == doc.metadata.source

    def test_add_document_with_embedding(self) -> None:
        """Test adding documents with embeddings."""
        memory = MemoryStore()

        # Create embedding using the new Generator API
        rng = np.random.default_rng()
        embedding = rng.random(768).astype(np.float32)

        # Add document with embedding
        doc = memory.add_document(
            content="Test document with embedding",
            metadata={"source": "test_embedding.txt"},
            embedding=embedding.tolist(),
        )

        # Verify document has embedding
        assert doc.embedding is not None
        assert np.array_equal(doc.embedding, embedding)

        # Test retrieval
        retrieved_doc = memory.get_document(doc.id)
        assert retrieved_doc is not None
        assert hasattr(retrieved_doc, "embedding") and retrieved_doc.embedding is not None
        assert np.array_equal(retrieved_doc.embedding, embedding)

    def test_search(self) -> None:
        """Test searching documents."""
        memory = MemoryStore()

        # Add documents
        docs = []
        for i in range(5):
            # Create embedding using the new Generator API
            rng = np.random.default_rng()
            embedding = rng.random(768).astype(np.float32)

            # Add document
            doc = memory.add_document(
                content=f"Test document {i}",
                metadata={"source": f"test{i}.txt"},
                embedding=embedding.tolist(),
            )
            docs.append(doc)

        # Search by embedding
        results = memory.search_by_embedding(docs[0].embedding, limit=3)
        assert len(results) == self.EXPECTED_COUNT_1
        assert results[0][0].id == docs[0].id  # First result should be the query document

        # Search by content
        with patch("saplings.memory.memory_store.get_embedding") as mock_get_embedding:
            # Mock the embedding function to return a fixed embedding
            mock_get_embedding.return_value = docs[0].embedding

            results = memory.search("Test document 0", limit=3)
            assert len(results) == self.EXPECTED_COUNT_1
            assert results[0][0].id == docs[0].id

    def test_delete_document(self) -> None:
        """Test deleting documents."""
        memory = MemoryStore()

        # Add a document
        doc = memory.add_document(
            content="Document to delete", metadata={"source": "delete_test.txt"}
        )

        # Verify document exists
        assert memory.get_document(doc.id) is not None

        # Delete document
        result = memory.delete_document(doc.id)
        assert result is True

        # Verify document is deleted
        assert memory.get_document(doc.id) is None

    def test_update_document(self) -> None:
        """Test updating documents."""
        memory = MemoryStore()

        # Add a document
        doc = memory.add_document(
            content="Original content", metadata={"source": "update_test.txt"}
        )

        # Update document
        updated_doc = Document(
            id=doc.id, content="Updated content", metadata=DocumentMetadata(source="updated.txt")
        )

        result = memory.update_document(updated_doc)
        assert result is True

        # Verify document is updated
        retrieved_doc = memory.get_document(doc.id)
        assert retrieved_doc is not None
        assert retrieved_doc.content == "Updated content"
        assert retrieved_doc.metadata.source == "updated.txt"

    def test_clear(self) -> None:
        """Test clearing the memory store."""
        memory = MemoryStore()

        # Add documents
        for i in range(5):
            memory.add_document(content=f"Test document {i}", metadata={"source": f"test{i}.txt"})

        # Verify documents exist
        assert memory.count() == 5

        # Clear memory
        memory.clear()

        # Verify documents are deleted
        assert memory.count() == 0

    def test_save_load(self) -> None:
        """Test saving and loading the memory store."""
        memory = MemoryStore()

        # Add documents
        docs = []
        for i in range(5):
            doc = memory.add_document(
                content=f"Test document {i}", metadata={"source": f"test{i}.txt"}
            )
            docs.append(doc)

        # Save memory store
        with tempfile.TemporaryDirectory() as tmpdir:
            memory.save(tmpdir)

            # Create new memory store and load
            new_memory = MemoryStore()
            new_memory.load(tmpdir)

            # Verify documents were loaded
            assert new_memory.count() == 5
            for doc in docs:
                loaded_doc = new_memory.get_document(doc.id)
                assert loaded_doc is not None
                assert loaded_doc.content == doc.content
                assert loaded_doc.metadata.source == doc.metadata.source

    def test_chunking(self) -> None:
        """Test document chunking."""
        # Create memory store with chunking configuration
        config = MemoryConfig(chunk_size=100, chunk_overlap=20)
        memory = MemoryStore(config=config)

        # Create a long document
        long_content = "This is a test document. " * 20  # ~400 characters

        # Add document
        doc = memory.add_document(content=long_content, metadata={"source": "long_doc.txt"})

        # Verify document was chunked
        assert hasattr(doc, "chunks")
        assert len(doc.chunks) > 1  # Should have multiple chunks

        # Verify chunk content
        total_content = ""
        for chunk in doc.chunks:
            assert len(chunk.content) <= config.chunk_size + config.chunk_overlap
            total_content += chunk.content

        # All content should be preserved (allowing for some whitespace differences)
        assert long_content.replace(" ", "") in total_content.replace(" ", "")
