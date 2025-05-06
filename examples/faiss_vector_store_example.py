#!/usr/bin/env python
"""
Example demonstrating the FAISS vector store integration in Saplings.

This example shows how to:
1. Configure and create a FAISS vector store
2. Add documents with embeddings
3. Perform similarity searches
4. Save and load the vector store
"""

from __future__ import annotations

import os

import numpy as np

from saplings.memory import Document, DocumentMetadata
from saplings.memory.config import MemoryConfig
from saplings.retrieval.faiss_vector_store import FaissVectorStore


def main():
    """Run the FAISS vector store example."""
    print("FAISS Vector Store Example")
    print("=========================")

    # Create a memory configuration with FAISS vector store
    print("\nCreating FAISS vector store configuration...")
    config = MemoryConfig.with_faiss(use_gpu=False)

    # Create the FAISS vector store
    print("Initializing FAISS vector store...")
    vector_store = FaissVectorStore(config)

    # Create some test documents with embeddings
    print("\nCreating test documents...")
    documents = create_test_documents(10, 768)  # 10 documents with 768-dim embeddings

    # Add documents to the vector store
    print(f"Adding {len(documents)} documents to vector store...")
    vector_store.add_documents(documents)

    # Perform a similarity search
    print("\nPerforming similarity search...")
    query_embedding = np.random.randn(768).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize

    results = vector_store.search(query_embedding, limit=3)

    print(f"Found {len(results)} similar documents:")
    for i, (doc, score) in enumerate(results):
        print(f"  {i+1}. Document: {doc.content[:50]}... (score: {score:.4f})")

    # Save the vector store
    output_dir = "faiss_example_output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving vector store to {output_dir}...")
    vector_store.save(output_dir)

    # Create a new vector store and load the saved data
    print("Loading vector store from disk...")
    new_vector_store = FaissVectorStore(config)
    new_vector_store.load(output_dir)

    # Verify the loaded store
    print(f"Loaded {len(new_vector_store.documents)} documents.")

    # Perform the same search and verify results
    print("\nPerforming search with loaded vector store...")
    new_results = new_vector_store.search(query_embedding, limit=3)

    print(f"Found {len(new_results)} similar documents:")
    for i, (doc, score) in enumerate(new_results):
        print(f"  {i+1}. Document: {doc.content[:50]}... (score: {score:.4f})")

    print("\nExample completed successfully!")


def create_test_documents(num_docs: int, embedding_dim: int) -> list[Document]:
    """
    Create test documents with random embeddings.

    Args:
    ----
        num_docs: Number of documents to create
        embedding_dim: Dimension of embeddings

    Returns:
    -------
        List[Document]: List of documents with embeddings

    """
    documents = []

    for i in range(num_docs):
        # Create a random embedding
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        # Create document
        content = (
            f"Test document {i+1} with some content for similarity testing. "
            f"This document contains information about topic {i % 3 + 1}."
        )

        doc = Document(
            content=content,
            metadata=DocumentMetadata(
                source=f"test_{i+1}.txt",
            ),
            embedding=embedding,
        )

        documents.append(doc)

    return documents


if __name__ == "__main__":
    main()
