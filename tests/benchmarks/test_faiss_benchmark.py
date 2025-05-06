from __future__ import annotations

"""
Benchmark tests for FAISS vector store.

These tests measure the performance of FAISS vector store compared to in-memory vector store.
They are marked as benchmark tests and are skipped by default.
"""


import time

import numpy as np
import pytest

from saplings.memory import Document, DocumentMetadata
from saplings.memory.config import MemoryConfig
from saplings.memory.vector_store import InMemoryVectorStore

# Try to import FAISS components
try:
    import faiss

    from saplings.retrieval.faiss_vector_store import FaissVectorStore

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Skip all tests if FAISS is not installed
pytestmark = [
    pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed"),
    pytest.mark.benchmark,  # Mark as benchmark test
]


def generate_test_documents(count: int, dim: int = 768) -> list[Document]:
    """Generate test documents with random embeddings."""
    docs = []
    for i in range(count):
        doc = Document(
            id=f"test{i}",
            content=f"Test document {i}",
            metadata=DocumentMetadata(source=f"test{i}.txt"),
            embedding=np.random.rand(dim).astype(np.float32),
        )
        docs.append(doc)
    return docs


def measure_insert_time(store, docs: list[Document]) -> float:
    """Measure time to insert documents into store."""
    start_time = time.time()
    store.add_documents(docs)
    return time.time() - start_time


def measure_query_time(
    store, query_embeddings: list[np.ndarray], top_k: int
) -> tuple[float, list[list[tuple[Document, float]]]]:
    """Measure time to query store."""
    results = []
    start_time = time.time()
    for embedding in query_embeddings:
        result = store.search(embedding, limit=top_k)
        results.append(result)
    query_time = time.time() - start_time
    return query_time, results


class TestFaissBenchmark:
    """Benchmark tests for FAISS vector store."""

    @pytest.mark.parametrize("doc_count", [100, 1000, 10000])
    def test_insertion_benchmark(self, doc_count) -> None:
        """Benchmark document insertion."""
        # Generate test documents
        docs = generate_test_documents(doc_count)

        # Configure stores
        config = MemoryConfig.with_faiss(use_gpu=False)

        # Create stores
        in_memory_store = InMemoryVectorStore(config)
        faiss_store = FaissVectorStore(config, use_gpu=False)

        # Measure insertion time
        measure_insert_time(in_memory_store, docs)
        measure_insert_time(faiss_store, docs)

        # Print results

        # No assertions, just performance measurement

    @pytest.mark.parametrize(
        ("doc_count", "query_count", "top_k"), [(1000, 10, 10), (10000, 10, 10), (10000, 10, 100)]
    )
    def test_query_benchmark(self, doc_count, query_count, top_k) -> None:
        """Benchmark query performance."""
        # Generate test documents
        docs = generate_test_documents(doc_count)

        # Generate query embeddings
        query_embeddings = [np.random.rand(768).astype(np.float32) for _ in range(query_count)]

        # Configure stores
        config = MemoryConfig.with_faiss(use_gpu=False)

        # Create stores
        in_memory_store = InMemoryVectorStore(config)
        faiss_store = FaissVectorStore(config, use_gpu=False)

        # Add documents
        in_memory_store.add_documents(docs)
        faiss_store.add_documents(docs)

        # Measure query time
        in_memory_time, in_memory_results = measure_query_time(
            in_memory_store, query_embeddings, top_k
        )
        faiss_time, faiss_results = measure_query_time(faiss_store, query_embeddings, top_k)

        # Print results

        # No assertions, just performance measurement

    @pytest.mark.parametrize("use_gpu", [False, True])
    def test_gpu_vs_cpu_benchmark(self, use_gpu) -> None:
        """Benchmark GPU vs CPU performance."""
        # Skip if GPU is requested but not available
        if use_gpu:
            try:
                # Check if FAISS was built with GPU support
                if not hasattr(faiss, "StandardGpuResources"):
                    pytest.skip("FAISS was not built with GPU support")

                # Try to create GPU resources
                faiss.StandardGpuResources()
            except Exception as e:
                pytest.skip(f"GPU not available: {e}")

        # Generate test documents
        doc_count = 10000
        docs = generate_test_documents(doc_count)

        # Generate query embeddings
        query_count = 10
        query_embeddings = [np.random.rand(768).astype(np.float32) for _ in range(query_count)]

        # Configure store
        config = MemoryConfig.with_faiss(use_gpu=use_gpu)

        # Create store
        faiss_store = FaissVectorStore(config, use_gpu=use_gpu)

        # Measure insertion time
        measure_insert_time(faiss_store, docs)

        # Measure query time
        query_time, _ = measure_query_time(faiss_store, query_embeddings, 10)

        # Print results

        # No assertions, just performance measurement
