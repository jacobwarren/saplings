#!/usr/bin/env python
"""
Benchmark comparing FAISS vector store performance against the original implementation.

This benchmark measures:
1. Memory usage
2. Query speed
3. Insert speed
4. Accuracy (recall)

Run with:
    python benchmarks/faiss_benchmark.py --dataset-sizes 1000 10000 100000 --embedding-dims 768 1536
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to path to allow importing saplings modules
sys.path.append(str(Path(__file__).parent.parent))

from saplings.memory import Document, DocumentMetadata, MemoryStore
from saplings.memory.config import MemoryConfig
from saplings.memory.vector_store import InMemoryVectorStore
from saplings.retrieval.faiss_vector_store import FaissVectorStore


def main():
    """Run the vector store benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark FAISS vector store performance")
    parser.add_argument(
        "--dataset-sizes",
        type=int,
        nargs="+",
        default=[1000, 10000, 100000],
        help="Dataset sizes to benchmark",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        nargs="+",
        default=[768, 1536],
        help="Embedding dimensions to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration for FAISS (if available)"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to run for each configuration",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to retrieve for each query"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run benchmarks for each dataset size and embedding dimension
    results = []
    for dataset_size in args.dataset_sizes:
        for embedding_dim in args.embedding_dims:
            print(f"\nBenchmarking dataset_size={dataset_size}, embedding_dim={embedding_dim}")

            result = benchmark_vector_stores(
                dataset_size=dataset_size,
                embedding_dim=embedding_dim,
                num_queries=args.num_queries,
                top_k=args.top_k,
                use_gpu=args.use_gpu,
            )
            results.append(result)

            # Free memory
            gc.collect()

    # Generate plots
    generate_plots(results, args.output_dir)

    print(f"\nBenchmark results saved to {args.output_dir}")


def benchmark_vector_stores(
    dataset_size: int,
    embedding_dim: int,
    num_queries: int = 100,
    top_k: int = 10,
    use_gpu: bool = False,
) -> Dict:
    """
    Benchmark original vs. FAISS vector stores.

    Args:
    ----
        dataset_size: Number of documents to insert
        embedding_dim: Dimension of embeddings
        num_queries: Number of queries to run
        top_k: Number of results to retrieve per query
        use_gpu: Whether to use GPU acceleration for FAISS

    Returns:
    -------
        Dict: Benchmark results

    """
    # Generate synthetic documents and embeddings
    docs, query_embeddings = generate_synthetic_data(dataset_size, embedding_dim, num_queries)

    # Configure stores
    config = MemoryConfig.with_faiss(use_gpu=use_gpu)

    # Original store
    start_time = time.time()
    original_store = InMemoryVectorStore(config)
    original_store_load_time = time.time() - start_time

    # FAISS store
    start_time = time.time()
    faiss_store = FaissVectorStore(config, use_gpu=use_gpu)
    faiss_store_load_time = time.time() - start_time

    # Benchmark insertion
    original_insert_time = measure_insert_time(original_store, docs)
    faiss_insert_time = measure_insert_time(faiss_store, docs)

    # Benchmark query time
    original_query_time, original_results = measure_query_time(
        original_store, query_embeddings, top_k
    )

    faiss_query_time, faiss_results = measure_query_time(faiss_store, query_embeddings, top_k)

    # Measure recall (accuracy)
    recall = calculate_recall(original_results, faiss_results)

    # Prepare results
    result = {
        "dataset_size": dataset_size,
        "embedding_dim": embedding_dim,
        "use_gpu": use_gpu,
        "original_store": {
            "load_time": original_store_load_time,
            "insert_time": original_insert_time,
            "query_time": original_query_time,
        },
        "faiss_store": {
            "load_time": faiss_store_load_time,
            "insert_time": faiss_insert_time,
            "query_time": faiss_query_time,
        },
        "recall": recall,
        "speedup": {
            "load": original_store_load_time / faiss_store_load_time
            if faiss_store_load_time > 0
            else float("inf"),
            "insert": original_insert_time / faiss_insert_time
            if faiss_insert_time > 0
            else float("inf"),
            "query": original_query_time / faiss_query_time
            if faiss_query_time > 0
            else float("inf"),
        },
    }

    # Print results
    print(f"Results for dataset_size={dataset_size}, embedding_dim={embedding_dim}:")
    print(f"  Original store load time: {original_store_load_time:.4f}s")
    print(f"  FAISS store load time: {faiss_store_load_time:.4f}s")
    print(f"  Original store insert time: {original_insert_time:.4f}s")
    print(f"  FAISS store insert time: {faiss_insert_time:.4f}s")
    print(f"  Original store query time: {original_query_time:.4f}s")
    print(f"  FAISS store query time: {faiss_query_time:.4f}s")
    print(f"  Recall: {recall:.4f}")
    print(f"  Speedup (load): {result['speedup']['load']:.2f}x")
    print(f"  Speedup (insert): {result['speedup']['insert']:.2f}x")
    print(f"  Speedup (query): {result['speedup']['query']:.2f}x")

    return result


def generate_synthetic_data(
    dataset_size: int, embedding_dim: int, num_queries: int
) -> Tuple[List[Document], List[np.ndarray]]:
    """
    Generate synthetic documents and query embeddings.

    Args:
    ----
        dataset_size: Number of documents to generate
        embedding_dim: Dimension of embeddings
        num_queries: Number of query embeddings to generate

    Returns:
    -------
        Tuple[List[Document], List[np.ndarray]]: Documents and query embeddings

    """
    print(f"Generating {dataset_size} documents with {embedding_dim}-dim embeddings...")

    documents = []
    for i in range(dataset_size):
        # Create a random embedding
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        # Create document
        content = f"Document {i+1} with synthetic content for benchmark testing."
        doc = Document(
            content=content,
            metadata=DocumentMetadata(source=f"synthetic_{i+1}.txt"),
            embedding=embedding,
        )

        documents.append(doc)

    # Generate query embeddings
    query_embeddings = []
    for i in range(num_queries):
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        query_embeddings.append(embedding)

    return documents, query_embeddings


def measure_insert_time(store: MemoryStore, docs: List[Document]) -> float:
    """
    Measure time to insert documents into a vector store.

    Args:
    ----
        store: Vector store
        docs: Documents to insert

    Returns:
    -------
        float: Insertion time in seconds

    """
    gc.collect()  # Clear memory

    start_time = time.time()
    store.add_documents(docs)
    insert_time = time.time() - start_time

    return insert_time


def measure_query_time(
    store: MemoryStore, query_embeddings: List[np.ndarray], top_k: int
) -> Tuple[float, List[List[str]]]:
    """
    Measure query time for a vector store.

    Args:
    ----
        store: Vector store
        query_embeddings: Query embeddings
        top_k: Number of results to retrieve per query

    Returns:
    -------
        Tuple[float, List[List[str]]]: Query time and results

    """
    gc.collect()  # Clear memory

    results = []

    start_time = time.time()
    for embedding in query_embeddings:
        # Retrieve similar documents
        similar_docs = store.similar_by_vector(embedding, limit=top_k)
        results.append([doc.id for doc, _ in similar_docs])

    query_time = (time.time() - start_time) / len(query_embeddings)

    return query_time, results


def calculate_recall(ground_truth: List[List[str]], results: List[List[str]]) -> float:
    """
    Calculate recall between ground truth and FAISS results.

    Args:
    ----
        ground_truth: Original results (list of lists of document IDs)
        results: FAISS results (list of lists of document IDs)

    Returns:
    -------
        float: Recall (0-1)

    """
    if not ground_truth or not results:
        return 0.0

    total_recall = 0.0
    for gt_ids, result_ids in zip(ground_truth, results):
        if not gt_ids:
            continue

        # Convert to sets for intersection
        gt_set = set(gt_ids)
        result_set = set(result_ids)

        # Calculate recall for this query
        intersection = len(gt_set.intersection(result_set))
        recall = intersection / len(gt_set) if gt_set else 1.0

        total_recall += recall

    # Average recall across all queries
    avg_recall = total_recall / len(ground_truth) if ground_truth else 0.0

    return avg_recall


def generate_plots(results: List[Dict], output_dir: str):
    """
    Generate plots from benchmark results.

    Args:
    ----
        results: List of benchmark results
        output_dir: Directory to save plots

    """
    # Prepare data for plots
    dataset_sizes = sorted(list(set(r["dataset_size"] for r in results)))
    embedding_dims = sorted(list(set(r["embedding_dim"] for r in results)))

    for embedding_dim in embedding_dims:
        # Filter results for this embedding dimension
        dim_results = [r for r in results if r["embedding_dim"] == embedding_dim]

        # Sort by dataset size
        dim_results.sort(key=lambda r: r["dataset_size"])

        # Extract data for plots
        sizes = [r["dataset_size"] for r in dim_results]
        original_query_times = [r["original_store"]["query_time"] for r in dim_results]
        faiss_query_times = [r["faiss_store"]["query_time"] for r in dim_results]
        original_insert_times = [r["original_store"]["insert_time"] for r in dim_results]
        faiss_insert_times = [r["faiss_store"]["insert_time"] for r in dim_results]
        recalls = [r["recall"] for r in dim_results]
        query_speedups = [r["speedup"]["query"] for r in dim_results]
        insert_speedups = [r["speedup"]["insert"] for r in dim_results]

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"FAISS vs Original Vector Store (dim={embedding_dim})")

        # Query time plot
        ax = axes[0, 0]
        ax.plot(sizes, original_query_times, "o-", label="Original")
        ax.plot(sizes, faiss_query_times, "o-", label="FAISS")
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Query Time (s)")
        ax.set_title("Query Time")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Insert time plot
        ax = axes[0, 1]
        ax.plot(sizes, original_insert_times, "o-", label="Original")
        ax.plot(sizes, faiss_insert_times, "o-", label="FAISS")
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Insert Time (s)")
        ax.set_title("Insert Time")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Speedup plot
        ax = axes[1, 0]
        ax.plot(sizes, query_speedups, "o-", label="Query Speedup")
        ax.plot(sizes, insert_speedups, "o-", label="Insert Speedup")
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Speedup (x)")
        ax.set_title("Speedup (Original/FAISS)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Recall plot
        ax = axes[1, 1]
        ax.plot(sizes, recalls, "o-")
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Recall")
        ax.set_title("Recall (higher is better)")
        ax.set_xscale("log")
        ax.axhline(y=0.99, linestyle="--", color="red", alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"faiss_benchmark_dim{embedding_dim}.png"))
        plt.close()

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by embedding dimension and compute average speedup
    for embedding_dim in embedding_dims:
        dim_results = [r for r in results if r["embedding_dim"] == embedding_dim]
        dim_results.sort(key=lambda r: r["dataset_size"])

        sizes = [r["dataset_size"] for r in dim_results]
        speedups = [r["speedup"]["query"] for r in dim_results]

        ax.plot(sizes, speedups, "o-", label=f"dim={embedding_dim}")

    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Query Speedup (x)")
    ax.set_title("FAISS Query Speedup by Dimension")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "faiss_speedup_summary.png"))
    plt.close()


if __name__ == "__main__":
    main()
