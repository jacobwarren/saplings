"""
Tests for the vector_store module.
"""

import os
import tempfile
from typing import List

import numpy as np
import pytest

from saplings.memory.config import MemoryConfig, SimilarityMetric, VectorStoreType
from saplings.memory.document import Document, DocumentMetadata
from saplings.memory.vector_store import InMemoryVectorStore, VectorStore, get_vector_store


class TestInMemoryVectorStore:
    """Tests for the InMemoryVectorStore class."""

    def setup_method(self):
        """Set up test environment."""
        self.config = MemoryConfig(
            vector_store={
                "store_type": VectorStoreType.IN_MEMORY,
                "similarity_metric": SimilarityMetric.COSINE,
            }
        )
        self.store = InMemoryVectorStore(self.config)

        # Create test documents
        self.docs = [
            Document(
                id="doc1",
                content="This is document 1.",
                metadata=DocumentMetadata(source="test1.txt"),
                embedding=np.array([1.0, 0.0, 0.0]),
            ),
            Document(
                id="doc2",
                content="This is document 2.",
                metadata=DocumentMetadata(source="test2.txt"),
                embedding=np.array([0.0, 1.0, 0.0]),
            ),
            Document(
                id="doc3",
                content="This is document 3.",
                metadata=DocumentMetadata(source="test3.txt"),
                embedding=np.array([0.0, 0.0, 1.0]),
            ),
        ]

    def test_add_document(self):
        """Test adding a document."""
        self.store.add_document(self.docs[0])

        assert len(self.store.documents) == 1
        assert len(self.store.embeddings) == 1
        assert self.store.documents["doc1"] == self.docs[0]
        assert np.array_equal(self.store.embeddings["doc1"], self.docs[0].embedding)

    def test_add_document_without_embedding(self):
        """Test adding a document without an embedding."""
        doc = Document(
            id="doc_no_embedding",
            content="This document has no embedding.",
            metadata=DocumentMetadata(source="test.txt"),
        )

        with pytest.raises(ValueError):
            self.store.add_document(doc)

    def test_add_documents(self):
        """Test adding multiple documents."""
        self.store.add_documents(self.docs)

        assert len(self.store.documents) == 3
        assert len(self.store.embeddings) == 3

        for doc in self.docs:
            assert self.store.documents[doc.id] == doc
            assert np.array_equal(self.store.embeddings[doc.id], doc.embedding)

    def test_search_cosine(self):
        """Test searching with cosine similarity."""
        self.store.add_documents(self.docs)

        # Search for a vector similar to doc1
        query = np.array([0.9, 0.1, 0.0])
        results = self.store.search(query, limit=2)

        assert len(results) == 2
        assert results[0][0].id == "doc1"
        assert results[1][0].id == "doc2"
        assert results[0][1] > results[1][1]  # doc1 should have higher similarity

    def test_search_dot_product(self):
        """Test searching with dot product similarity."""
        self.config.vector_store.similarity_metric = SimilarityMetric.DOT_PRODUCT
        self.store = InMemoryVectorStore(self.config)
        self.store.add_documents(self.docs)

        # Search for a vector similar to doc1
        query = np.array([0.9, 0.1, 0.0])
        results = self.store.search(query, limit=2)

        assert len(results) == 2
        assert results[0][0].id == "doc1"
        assert results[1][0].id == "doc2"

    def test_search_euclidean(self):
        """Test searching with euclidean similarity."""
        self.config.vector_store.similarity_metric = SimilarityMetric.EUCLIDEAN
        self.store = InMemoryVectorStore(self.config)
        self.store.add_documents(self.docs)

        # Search for a vector similar to doc1
        query = np.array([0.9, 0.1, 0.0])
        results = self.store.search(query, limit=2)

        assert len(results) == 2
        assert results[0][0].id == "doc1"
        assert results[1][0].id == "doc2"

    def test_search_with_filter(self):
        """Test searching with a filter."""
        self.store.add_documents(self.docs)

        # Search with a filter on metadata.source
        query = np.array([0.5, 0.5, 0.5])
        results = self.store.search(query, limit=3, filter_dict={"metadata.source": "test2.txt"})

        assert len(results) == 1
        assert results[0][0].id == "doc2"

    def test_delete(self):
        """Test deleting a document."""
        self.store.add_documents(self.docs)

        # Delete a document
        result = self.store.delete("doc2")

        assert result is True
        assert len(self.store.documents) == 2
        assert len(self.store.embeddings) == 2
        assert "doc2" not in self.store.documents
        assert "doc2" not in self.store.embeddings

        # Try to delete a non-existent document
        result = self.store.delete("non_existent")

        assert result is False

    def test_update(self):
        """Test updating a document."""
        self.store.add_documents(self.docs)

        # Update a document
        updated_doc = Document(
            id="doc2",
            content="This is updated document 2.",
            metadata=DocumentMetadata(source="updated.txt"),
            embedding=np.array([0.5, 0.5, 0.0]),
        )

        self.store.update(updated_doc)

        assert self.store.documents["doc2"] == updated_doc
        assert np.array_equal(self.store.embeddings["doc2"], updated_doc.embedding)

        # Try to update a non-existent document
        non_existent_doc = Document(
            id="non_existent",
            content="This document doesn't exist.",
            metadata=DocumentMetadata(source="test.txt"),
            embedding=np.array([0.0, 0.0, 0.0]),
        )

        with pytest.raises(ValueError):
            self.store.update(non_existent_doc)

    def test_get(self):
        """Test getting a document."""
        self.store.add_documents(self.docs)

        # Get an existing document
        doc = self.store.get("doc2")

        assert doc == self.docs[1]

        # Try to get a non-existent document
        doc = self.store.get("non_existent")

        assert doc is None

    def test_list(self):
        """Test listing documents."""
        self.store.add_documents(self.docs)

        # List all documents
        docs = self.store.list()

        assert len(docs) == 3
        assert set(doc.id for doc in docs) == {"doc1", "doc2", "doc3"}

        # List with a limit
        docs = self.store.list(limit=2)

        assert len(docs) == 2

        # List with a filter
        docs = self.store.list(filter_dict={"metadata.source": "test1.txt"})

        assert len(docs) == 1
        assert docs[0].id == "doc1"

    def test_count(self):
        """Test counting documents."""
        self.store.add_documents(self.docs)

        # Count all documents
        count = self.store.count()

        assert count == 3

        # Count with a filter
        count = self.store.count(filter_dict={"metadata.source": "test1.txt"})

        assert count == 1

    def test_clear(self):
        """Test clearing the store."""
        self.store.add_documents(self.docs)

        # Clear the store
        self.store.clear()

        assert len(self.store.documents) == 0
        assert len(self.store.embeddings) == 0

    def test_save_and_load(self):
        """Test saving and loading the store."""
        self.store.add_documents(self.docs)

        # Save the store
        with tempfile.TemporaryDirectory() as temp_dir:
            self.store.save(temp_dir)

            # Create a new store and load the saved data
            new_store = InMemoryVectorStore(self.config)
            new_store.load(temp_dir)

            # Check that the loaded store has the same documents
            assert len(new_store.documents) == 3
            assert len(new_store.embeddings) == 3

            for doc_id, doc in self.store.documents.items():
                assert doc_id in new_store.documents
                assert new_store.documents[doc_id].id == doc.id
                assert new_store.documents[doc_id].content == doc.content
                assert np.array_equal(new_store.embeddings[doc_id], self.store.embeddings[doc_id])


class TestGetVectorStore:
    """Tests for the get_vector_store function."""

    def test_get_in_memory_store(self):
        """Test getting an in-memory vector store."""
        config = MemoryConfig(vector_store={"store_type": VectorStoreType.IN_MEMORY})
        store = get_vector_store(config)

        assert isinstance(store, InMemoryVectorStore)

    def test_get_unsupported_store(self):
        """Test getting an unsupported vector store."""
        config = MemoryConfig(vector_store={"store_type": VectorStoreType.FAISS})

        with pytest.raises(ValueError):
            get_vector_store(config)

    def test_get_default_store(self):
        """Test getting the default vector store."""
        store = get_vector_store()

        assert isinstance(store, InMemoryVectorStore)
