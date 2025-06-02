from __future__ import annotations

"""
Unit tests for vector stores.
"""


import tempfile

import numpy as np

from saplings.memory import Document, DocumentMetadata
from saplings.memory.config import MemoryConfig, SimilarityMetric
from saplings.memory.vector_store import InMemoryVectorStore


class TestInMemoryVectorStore:
    EXPECTED_COUNT_1 = 5

    """Test the in-memory vector store."""

    def test_initialization(self) -> None:
        """Test vector store initialization."""
        # Test with default configuration
        store = InMemoryVectorStore()
        assert store.config is not None

        # Test with custom configuration
        config = MemoryConfig.default()
        # Modify the similarity metric
        config.vector_store.similarity_metric = SimilarityMetric.COSINE
        store = InMemoryVectorStore(config=config)
        assert store.similarity_metric == SimilarityMetric.COSINE

    def test_add_document(self) -> None:
        """Test adding documents to vector store."""
        store = InMemoryVectorStore()

        # Create a document with embedding
        doc = Document(
            id="test1",
            content="Test content",
            metadata=DocumentMetadata(
                source="test.txt", content_type="text", language="en", author="tester"
            ),
            embedding=np.random.default_rng().random(768).astype(np.float32).tolist(),
        )

        # Add document
        store.add_document(doc)

        # Verify document was added
        assert store.count() == 1
        assert "test1" in store.documents

        # Test retrieval
        retrieved_doc = store.get("test1")
        assert retrieved_doc is not None
        assert retrieved_doc.id == doc.id
        assert retrieved_doc.content == doc.content
        assert retrieved_doc.embedding is not None
        assert doc.embedding is not None
        assert np.array_equal(retrieved_doc.embedding, doc.embedding)

    def test_add_documents(self) -> None:
        """Test adding multiple documents."""
        store = InMemoryVectorStore()

        # Create documents
        docs = []
        for i in range(5):
            doc = Document(
                id=f"test{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(
                    source=f"test{i}.txt", content_type="text", language="en", author="tester"
                ),
                embedding=np.random.default_rng().random(768).astype(np.float32).tolist(),
            )
            docs.append(doc)

        # Add documents
        store.add_documents(docs)

        # Verify documents were added
        assert store.count() == 5
        for i in range(5):
            assert f"test{i}" in store.documents

    def test_search(self) -> None:
        """Test searching documents."""
        store = InMemoryVectorStore()

        # Create documents with embeddings
        docs = []
        for i in range(10):
            # Create embedding
            embedding = np.random.default_rng().random(768).astype(np.float32)

            # Add document
            doc = Document(
                id=f"test{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(
                    source=f"test{i}.txt", content_type="text", language="en", author="tester"
                ),
                embedding=embedding.tolist(),
            )
            docs.append(doc)
            store.add_document(doc)

        # Search by embedding
        results = store.search(docs[0].embedding, limit=5)
        assert len(results) == self.EXPECTED_COUNT_1
        assert results[0][0].id == docs[0].id  # First result should be the query document

        # Test with filter using metadata prefix
        results = store.search(
            docs[0].embedding, limit=5, filter_dict={"metadata.source": "test1.txt"}
        )
        assert len(results) == 1
        assert results[0][0].id == "test1"

    def test_delete(self) -> None:
        """Test deleting documents."""
        store = InMemoryVectorStore()

        # Create and add document
        doc = Document(
            id="test_delete",
            content="Document to delete",
            metadata=DocumentMetadata(
                source="delete_test.txt", content_type="text", language="en", author="tester"
            ),
            embedding=np.random.default_rng().random(768).astype(np.float32).tolist(),
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
        """Test updating documents."""
        store = InMemoryVectorStore()

        # Create and add document
        doc = Document(
            id="test_update",
            content="Original content",
            metadata=DocumentMetadata(
                source="update_test.txt", content_type="text", language="en", author="tester"
            ),
            embedding=np.random.default_rng().random(768).astype(np.float32).tolist(),
        )
        store.add_document(doc)

        # Update document
        updated_doc = Document(
            id="test_update",
            content="Updated content",
            metadata=DocumentMetadata(
                source="updated.txt", content_type="text", language="en", author="tester"
            ),
            embedding=np.random.default_rng().random(768).astype(np.float32).tolist(),
        )

        store.update(updated_doc)

        # Verify document is updated
        retrieved_doc = store.get("test_update")
        assert retrieved_doc is not None
        assert retrieved_doc.content == "Updated content"
        assert isinstance(retrieved_doc.metadata, DocumentMetadata)
        assert retrieved_doc.metadata.source == "updated.txt"
        assert retrieved_doc.embedding is not None
        assert updated_doc.embedding is not None
        assert np.array_equal(retrieved_doc.embedding, updated_doc.embedding)

    def test_list(self) -> None:
        """Test listing documents."""
        store = InMemoryVectorStore()

        # Add documents
        for i in range(10):
            doc = Document(
                id=f"test{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(
                    source=f"test{i}.txt", content_type="text", language="en", author="tester"
                ),
                embedding=np.random.default_rng().random(768).astype(np.float32).tolist(),
            )
            store.add_document(doc)

        # List all documents
        docs = store.list()
        assert len(docs) == 10

        # List with limit
        docs = store.list(limit=5)
        assert len(docs) == self.EXPECTED_COUNT_1

        # List with filter using metadata prefix
        docs = store.list(filter_dict={"metadata.source": "test1.txt"})
        assert len(docs) == 1
        assert docs[0].id == "test1"

    def test_count(self) -> None:
        """Test counting documents."""
        store = InMemoryVectorStore()

        # Add documents
        for i in range(10):
            doc = Document(
                id=f"test{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(
                    source=f"test{i}.txt", content_type="text", language="en", author="tester"
                ),
                embedding=np.random.default_rng().random(768).astype(np.float32).tolist(),
            )
            store.add_document(doc)

        # Count all documents
        count = store.count()
        assert count == 10

        # Count with filter using metadata prefix
        count = store.count(filter_dict={"metadata.source": "test1.txt"})
        assert count == 1

    def test_clear(self) -> None:
        """Test clearing the vector store."""
        store = InMemoryVectorStore()

        # Add documents
        for i in range(5):
            doc = Document(
                id=f"test{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(
                    source=f"test{i}.txt", content_type="text", language="en", author="tester"
                ),
                embedding=np.random.default_rng().random(768).astype(np.float32).tolist(),
            )
            store.add_document(doc)

        # Verify documents exist
        assert store.count() == 5

        # Clear store
        store.clear()

        # Verify documents are deleted
        assert store.count() == 0

    def test_save_load(self) -> None:
        """Test saving and loading the vector store."""
        store = InMemoryVectorStore()

        # Add documents
        docs = []
        for i in range(5):
            doc = Document(
                id=f"test{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(
                    source=f"test{i}.txt", content_type="text", language="en", author="tester"
                ),
                embedding=np.random.default_rng().random(768).astype(np.float32).tolist(),
            )
            docs.append(doc)
            store.add_document(doc)

        # Save vector store
        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)

            # Create new vector store and load
            new_store = InMemoryVectorStore()
            new_store.load(tmpdir)

            # Verify documents were loaded
            assert new_store.count() == 5
            for doc in docs:
                loaded_doc = new_store.get(doc.id)
                assert loaded_doc is not None
                assert loaded_doc.content == doc.content
                assert isinstance(loaded_doc.metadata, DocumentMetadata)
                assert isinstance(doc.metadata, DocumentMetadata)
                assert loaded_doc.metadata.source == doc.metadata.source
                assert loaded_doc.embedding is not None
                assert doc.embedding is not None
                assert np.array_equal(loaded_doc.embedding, doc.embedding)
