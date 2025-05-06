#!/usr/bin/env python
"""
Example demonstrating the optimized block-diagonal packing for GASA.

This example shows how to:
1. Use the PyTorch-based block packer for improved performance
2. Compare the performance with the original implementation
3. Visualize the resulting attention masks
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available, will only demonstrate the Python implementation")

from saplings.gasa.config import GASAConfig
from saplings.gasa.packing.block_diagonal_packer import BlockDiagonalPacker
from saplings.memory import DependencyGraph, Document, DocumentMetadata


def main():
    """Run the optimized block packing example."""
    print("Optimized GASA Block Packing Example")
    print("===================================")

    if not HAS_TORCH:
        print("\nWarning: PyTorch not available. Install PyTorch to see the full example.")
        print("Continuing with the Python implementation only...\n")

    # Create a simple document collection and graph
    print("Creating test documents and graph...")
    documents, graph = create_test_documents_and_graph(10)

    # Create a GASA config
    config = GASAConfig(max_hops=2)

    # Create the original block packer
    original_packer = BlockDiagonalPacker(graph, config)

    # Test the original implementation
    print("\nTesting original Python implementation...")
    start_time = time.time()

    # Create a sample attention mask for each document
    masks = []
    for doc in documents:
        # Simple 10x10 mask for demonstration
        mask = np.ones((10, 10), dtype=np.float32)
        masks.append(mask)

    # Pack the masks
    result_original = block_pack_original(masks)
    original_time = time.time() - start_time

    print(f"Original implementation took {original_time:.4f} seconds")
    print(f"Result shape: {result_original.shape}")

    # If PyTorch is available, test the optimized implementation
    if HAS_TORCH:
        from saplings.gasa.packing.block_pack import block_pack

        print("\nTesting PyTorch implementation...")

        # Convert masks to PyTorch tensors
        torch_masks = [torch.tensor(mask) for mask in masks]

        # Measure performance
        start_time = time.time()
        result_torch = block_pack(torch_masks)
        torch_time = time.time() - start_time

        # Convert to numpy for comparison
        result_torch_np = result_torch.cpu().numpy()

        print(f"PyTorch implementation took {torch_time:.4f} seconds")
        print(f"Result shape: {result_torch.shape}")

        # Calculate speedup
        speedup = original_time / torch_time
        print(f"Speedup: {speedup:.2f}x faster")

        # Verify results match
        if np.allclose(result_original, result_torch_np):
            print("Results match! ✓")
        else:
            print("Warning: Results don't match exactly")

        # Plot the results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(result_original, cmap="viridis")
        axes[0].set_title("Original Implementation")

        axes[1].imshow(result_torch_np, cmap="viridis")
        axes[1].set_title("PyTorch Implementation")

        plt.tight_layout()
        plt.savefig("block_packing_comparison.png")
        print("\nSaved visualization to block_packing_comparison.png")

    print("\nExample completed successfully!")


def block_pack_original(mask_list):
    """
    Original Python implementation of block packing.

    Args:
    ----
        mask_list: List of mask matrices

    Returns:
    -------
        np.ndarray: Block diagonal matrix

    """
    if not mask_list:
        return np.empty((0, 0))

    # Determine the total size
    total_rows = sum(mask.shape[0] for mask in mask_list)
    total_cols = sum(mask.shape[1] for mask in mask_list)

    # Create the output matrix
    result = np.zeros((total_rows, total_cols), dtype=mask_list[0].dtype)

    # Fill in the blocks
    row_offset = 0
    col_offset = 0

    for mask in mask_list:
        rows, cols = mask.shape
        result[row_offset : row_offset + rows, col_offset : col_offset + cols] = mask
        row_offset += rows
        col_offset += cols

    return result


def create_test_documents_and_graph(num_docs):
    """
    Create test documents and a dependency graph.

    Args:
    ----
        num_docs: Number of documents to create

    Returns:
    -------
        Tuple[List[Document], DependencyGraph]: Documents and graph

    """
    documents = []
    graph = DependencyGraph()

    # Create documents
    for i in range(num_docs):
        doc = Document(
            content=f"Test document {i+1} with some content for testing.",
            metadata=DocumentMetadata(
                source=f"test_{i+1}.txt",
            ),
        )

        documents.append(doc)
        graph.add_document_node(doc)

    # Add relationships between documents
    for i in range(num_docs - 1):
        # Connect to next document
        graph.add_relationship(
            f"document:{documents[i].id}", f"document:{documents[i+1].id}", "references"
        )

        # Add some cross-connections for larger sets\1# Threshold for num docs\1NUM_DOCS_THRESHOLD = 5\1if num_docs > NUM_DOCS_THRESHOLD:
        for i in range(0, num_docs, 2):
            if i + 3 < num_docs:
                graph.add_relationship(
                    f"document:{documents[i].id}", f"document:{documents[i+3].id}", "related"
                )

    return documents, graph


if __name__ == "__main__":
    main()
