"""
Tests for the document module.
"""

import numpy as np
import pytest
from datetime import datetime

from saplings.memory.document import Document, DocumentMetadata


class TestDocumentMetadata:
    """Tests for the DocumentMetadata class."""
    
    def test_create_metadata(self):
        """Test creating metadata."""
        metadata = DocumentMetadata(source="test.txt")
        
        assert metadata.source == "test.txt"
        assert metadata.content_type == "text/plain"
        assert metadata.language is None
        assert metadata.author is None
        assert metadata.tags == []
        assert metadata.custom == {}
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)
    
    def test_update_metadata(self):
        """Test updating metadata."""
        metadata = DocumentMetadata(source="test.txt")
        
        # Update standard fields
        metadata.update(
            language="en",
            author="Test Author",
            tags=["test", "document"],
        )
        
        assert metadata.language == "en"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "document"]
        
        # Update custom fields
        metadata.update(
            custom_field="custom value",
            another_field=123,
        )
        
        assert metadata.custom["custom_field"] == "custom value"
        assert metadata.custom["another_field"] == 123


class TestDocument:
    """Tests for the Document class."""
    
    def test_create_document(self):
        """Test creating a document."""
        metadata = DocumentMetadata(source="test.txt")
        document = Document(
            id="doc1",
            content="This is a test document.",
            metadata=metadata,
        )
        
        assert document.id == "doc1"
        assert document.content == "This is a test document."
        assert document.metadata == metadata
        assert document.embedding is None
        assert document.chunks == []
    
    def test_create_document_with_dict_metadata(self):
        """Test creating a document with dictionary metadata."""
        document = Document(
            id="doc1",
            content="This is a test document.",
            metadata={"source": "test.txt", "author": "Test Author"},
        )
        
        assert document.id == "doc1"
        assert document.content == "This is a test document."
        assert isinstance(document.metadata, DocumentMetadata)
        assert document.metadata.source == "test.txt"
        assert document.metadata.author == "Test Author"
    
    def test_update_embedding(self):
        """Test updating a document's embedding."""
        document = Document(
            id="doc1",
            content="This is a test document.",
            metadata={"source": "test.txt"},
        )
        
        # Update with list
        document.update_embedding([1.0, 2.0, 3.0])
        assert isinstance(document.embedding, np.ndarray)
        assert document.embedding.tolist() == [1.0, 2.0, 3.0]
        
        # Update with numpy array
        document.update_embedding(np.array([4.0, 5.0, 6.0]))
        assert document.embedding.tolist() == [4.0, 5.0, 6.0]
    
    def test_chunk_document(self):
        """Test chunking a document."""
        document = Document(
            id="doc1",
            content="This is a test document. It has multiple sentences. "
                   "We will use it to test the chunking functionality.",
            metadata={"source": "test.txt"},
        )
        
        # Chunk with size larger than content
        chunks = document.chunk(chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].id == "doc1"
        
        # Chunk with smaller size
        chunks = document.chunk(chunk_size=30)
        assert len(chunks) > 1
        
        # Check that chunks are stored in the document
        assert document.chunks == chunks
        
        # Check chunk content
        assert chunks[0].content.startswith("This is a test document.")
        
        # Check chunk metadata
        assert chunks[0].metadata.source == "test.txt"
        assert "parent_id" in chunks[0].metadata.custom
        assert chunks[0].metadata.custom["parent_id"] == "doc1"
    
    def test_to_dict_and_from_dict(self):
        """Test converting a document to and from a dictionary."""
        document = Document(
            id="doc1",
            content="This is a test document.",
            metadata={"source": "test.txt", "author": "Test Author"},
            embedding=np.array([1.0, 2.0, 3.0]),
        )
        
        # Convert to dictionary
        doc_dict = document.to_dict()
        
        assert doc_dict["id"] == "doc1"
        assert doc_dict["content"] == "This is a test document."
        assert doc_dict["metadata"]["source"] == "test.txt"
        assert doc_dict["metadata"]["author"] == "Test Author"
        assert doc_dict["embedding"] == [1.0, 2.0, 3.0]
        
        # Convert back to document
        new_document = Document.from_dict(doc_dict)
        
        assert new_document.id == document.id
        assert new_document.content == document.content
        assert new_document.metadata.source == document.metadata.source
        assert new_document.metadata.author == document.metadata.author
        assert new_document.embedding.tolist() == document.embedding.tolist()
