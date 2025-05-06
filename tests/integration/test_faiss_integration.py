from __future__ import annotations

"""
Integration tests for FAISS vector store.

These tests require FAISS to be installed and will be skipped if it's not available.
"""


import tempfile

import numpy as np
import pytest

from saplings.memory import Document, DocumentMetadata, MemoryStore
from saplings.memory.config import MemoryConfig

# Try to import FAISS components
try:
    import faiss

    from saplings.retrieval.faiss_vector_store import FaissVectorStore

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Skip all tests if FAISS is not installed
pytestmark = pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")


class TestFaissVectorStore:
    EXPECTED_COUNT_1 = 5

    """Test the FAISS vector store."""

    def test_initialization(self) -> None:
        """Test FAISS vector store initialization."""
        # Test with default configuration
        config = MemoryConfig.with_faiss(use_gpu=False)
        store = FaissVectorStore(config)
        assert store.config is not None
        assert store.use_gpu is False

        # Test with GPU configuration (will fall back to CPU if GPU not available)
        config = MemoryConfig.with_faiss(use_gpu=True)
        store = FaissVectorStore(config, use_gpu=True)
        # Note: We don't assert use_gpu here because it might be automatically disabled if GPU is not available

    def test_add_document(self) -> None:
        """Test adding documents to FAISS vector store."""
        config = MemoryConfig.with_faiss(use_gpu=False)
        store = FaissVectorStore(config)

        # Create a document with embedding
        doc = Document(
            id="test1",
            content="Test content",
            metadata=DocumentMetadata(source="test.txt"),
            embedding=np.random.rand(768).astype(np.float32),
        )

        # Add document
        store.add_document(doc)

        # Verify document was added
        assert store.count() == 1
        assert "test1" in store.documents

        # Test retrieval
        retrieved_doc = store.get("test1")
        assert retrieved_doc.id == doc.id
        assert retrieved_doc.content == doc.content
        assert np.array_equal(retrieved_doc.embedding, doc.embedding)

    def test_add_documents(self) -> None:
        """Test adding multiple documents to FAISS vector store."""
        config = MemoryConfig.with_faiss(use_gpu=False)
        store = FaissVectorStore(config)

        # Create documents
        docs = []
        for i in range(5):
            doc = Document(
                id=f"test{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(source=f"test{i}.txt"),
                embedding=np.random.rand(768).astype(np.float32),
            )
            docs.append(doc)

        # Add documents
        store.add_documents(docs)

        # Verify documents were added
        assert store.count() == 5
        for i in range(5):
            assert f"test{i}" in store.documents

    def test_search(self) -> None:
        """Test searching documents in FAISS vector store."""
        config = MemoryConfig.with_faiss(use_gpu=False)
        store = FaissVectorStore(config)

        # Create documents with embeddings
        docs = []
        for i in range(10):
            # Create embedding
            embedding = np.random.rand(768).astype(np.float32)

            # Add document
            doc = Document(
                id=f"test{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(source=f"test{i}.txt"),
                embedding=embedding,
            )
            docs.append(doc)
            store.add_document(doc)

        # Search by embedding
        results = store.search(docs[0].embedding, limit=5)
        assert len(results) == self.EXPECTED_COUNT_1
        assert results[0][0].id == docs[0].id  # First result should be the query document

        # Test with filter
        results = store.search(docs[0].embedding, limit=5, filter_dict={"source": "test1.txt"})
        assert len(results) == 1
        assert results[0][0].id == "test1"

    def test_delete(self) -> None:
        """Test deleting documents from FAISS vector store."""
        config = MemoryConfig.with_faiss(use_gpu=False)
        store = FaissVectorStore(config)

        # Create and add document
        doc = Document(
            id="test_delete",
            content="Document to delete",
            metadata=DocumentMetadata(source="delete_test.txt"),
            embedding=np.random.rand(768).astype(np.float32),
        )
        store.add_document(doc)

        # Verify document exists
        assert store.get("test_delete") is not None

        # Delete document
        result = store.delete("test_delete")
        assert result is True

        # Verify document is deleted
        assert store.get("test_delete") is None

        # Test deleting non-existent document
        result = store.delete("non_existent")
        assert result is False

    def test_update(self) -> None:
        """Test updating documents in FAISS vector store."""
        config = MemoryConfig.with_faiss(use_gpu=False)
        store = FaissVectorStore(config)

        # Create and add document
        doc = Document(
            id="test_update",
            content="Original content",
            metadata=DocumentMetadata(source="update_test.txt"),
            embedding=np.random.rand(768).astype(np.float32),
        )
        store.add_document(doc)

        # Update document
        updated_doc = Document(
            id="test_update",
            content="Updated content",
            metadata=DocumentMetadata(source="updated.txt"),
            embedding=np.random.rand(768).astype(np.float32),
        )

        store.update(updated_doc)

        # Verify document is updated
        retrieved_doc = store.get("test_update")
        assert retrieved_doc.content == "Updated content"
        assert retrieved_doc.metadata.source == "updated.txt"
        assert np.array_equal(retrieved_doc.embedding, updated_doc.embedding)

    def test_save_load(self) -> None:
        """Test saving and loading the FAISS vector store."""
        config = MemoryConfig.with_faiss(use_gpu=False)
        store = FaissVectorStore(config)

        # Add documents
        docs = []
        for i in range(5):
            doc = Document(
                id=f"test{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(source=f"test{i}.txt"),
                embedding=np.random.rand(768).astype(np.float32),
            )
            docs.append(doc)
            store.add_document(doc)

        # Save vector store
        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)

            # Create new vector store and load
            new_store = FaissVectorStore(config)
            new_store.load(tmpdir)

            # Verify documents were loaded
            assert new_store.count() == 5
            for doc in docs:
                loaded_doc = new_store.get(doc.id)
                assert loaded_doc is not None
                assert loaded_doc.content == doc.content
                assert loaded_doc.metadata.source == doc.metadata.source
                assert np.array_equal(loaded_doc.embedding, doc.embedding)

    def test_memory_store_with_faiss(self) -> None:
        """Test memory store with FAISS vector store."""
        # Create memory store with FAISS
        config = MemoryConfig.with_faiss(use_gpu=False)
        memory = MemoryStore(config=config)

        # Verify vector store type
        assert isinstance(memory.vector_store, FaissVectorStore)

        # Add documents
        docs = []
        for i in range(10):
            doc = memory.add_document(
                content=f"Test document {i} about artificial intelligence.",
                metadata={"source": f"doc{i}.txt"},
                embedding=np.random.rand(768).astype(np.float32),
            )
            docs.append(doc)

        # Test search by embedding
        results = memory.search_by_embedding(docs[0].embedding, limit=5)
        assert len(results) == self.EXPECTED_COUNT_1
        assert results[0][0].id == docs[0].id  # First result should be the query document

        # Test save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            memory.save(tmpdir)

            # Create new memory store and load
            new_memory = MemoryStore(config=config)
            new_memory.load(tmpdir)

            # Verify documents were loaded
            assert new_memory.vector_store.count() == 10
            for doc in docs:
                loaded_doc = new_memory.get_document(doc.id)
                assert loaded_doc is not None
                assert loaded_doc.content == doc.content
                assert loaded_doc.metadata.source == doc.metadata.source
