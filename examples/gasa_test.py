"""
Test script for Graph-Aligned Sparse Attention (GASA).

This script demonstrates the improved GASA functionality, including:
1. Token mapping using exact positions during prompt composition
2. Efficient graph distance calculation with Dijkstra/Floyd-Warshall
3. Optimized sparse attention masks
4. Model adapter integration
5. Unified BlockDiagonalPacker implementation
"""

from __future__ import annotations

import argparse
import asyncio
import time

import matplotlib.pyplot as plt
import numpy as np

from saplings.core.model_adapter import LLM
from saplings.gasa.config import GASAConfig
from saplings.gasa.core.graph_distance import GraphDistanceCalculator
from saplings.gasa.mask_builder import MaskFormat, MaskType
from saplings.gasa.packing.block_diagonal_packer import BlockDiagonalPacker
from saplings.gasa.service.gasa_service import GASAService
from saplings.memory.document import Document, DocumentMetadata
from saplings.memory.graph import DependencyGraph


async def main():
    """Run GASA test."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="GASA Test")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3-8B", help="Model name (for tokenizer)"
    )
    parser.add_argument("--max-hops", type=int, default=2, help="Maximum hops for GASA")
    parser.add_argument("--visualize", action="store_true", help="Visualize attention masks")
    args = parser.parse_args()

    print(f"Testing GASA with max_hops={args.max_hops}")

    # Create model and get tokenizer
    model = LLM.create(provider="transformers", model=args.model)
    tokenizer = model.tokenizer

    # Create test documents
    documents = create_test_documents(5)

    # Create dependency graph
    graph = create_test_graph(documents)

    # Create GASA service
    config = GASAConfig(
        enabled=True,
        max_hops=args.max_hops,
        mask_strategy="binary",
        cache_masks=True,
    )
    gasa_service = GASAService(
        graph=graph,
        config=config,
        tokenizer=tokenizer,
    )

    # Create test prompt
    prompt = "Summarize the following documents:\n\n"
    for i, doc in enumerate(documents):
        prompt += f"Document {i+1}: {doc.content}\n\n"
    prompt += "Summary:"

    print(f"\nPrompt length: {len(prompt)} characters")

    # Test graph distance calculator
    start_time = time.time()
    graph_calc = GraphDistanceCalculator(graph)
    doc_ids = [f"doc_{i+1}" for i in range(len(documents))]
    distances = graph_calc.build_distance_matrix(doc_ids, args.max_hops)
    end_time = time.time()

    print(f"\nGraph distance computation time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Distance matrix entries: {len(distances)}")

    # Print some distance examples
    for i in range(min(3, len(documents))):
        for j in range(i + 1, min(i + 3, len(documents))):
            doc1_id = f"doc_{i+1}"
            doc2_id = f"doc_{j+1}"
            if (doc1_id, doc2_id) in distances:
                distance = distances[(doc1_id, doc2_id)]
                print(f"Distance from {doc1_id} to {doc2_id}: {distance}")

    # Build attention mask
    start_time = time.time()
    mask = gasa_service.build_mask(
        documents=documents,
        prompt=prompt,
        format=MaskFormat.DENSE,
        mask_type=MaskType.ATTENTION,
    )
    end_time = time.time()

    print(f"\nMask building time: {(end_time - start_time) * 1000:.2f} ms")

    # Calculate sparsity
    total_elements = mask.size
    nonzero_elements = np.count_nonzero(mask)
    zero_elements = total_elements - nonzero_elements
    sparsity = zero_elements / total_elements

    print(f"Mask shape: {mask.shape}")
    print(f"Mask sparsity: {sparsity:.2%}")
    print(f"Theoretical FLOP reduction: {sparsity:.2%}")

    # Test with TransformersAdapter
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()

    # Time the generation with sparse attention
    start_time = time.time()
    response = await model.generate(
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        attention_mask=mask,
    )
    end_time = time.time()

    print(f"\nGeneration with sparse attention: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Response: {response.text[:100]}...")

    # Test block diagonal packing
    start_time = time.time()
    packer = BlockDiagonalPacker(
        graph=graph,
        config=config,
        tokenizer=tokenizer,
    )
    reordered_input_ids, reordered_attention_mask, position_mapping = packer.reorder_tokens(
        documents=documents,
        prompt=prompt,
        input_ids=input_ids,
    )
    end_time = time.time()

    print(f"\nBlock diagonal packing time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Original sequence length: {len(input_ids)}")
    print(f"Reordered sequence length: {len(reordered_input_ids)}")

    # Visualize masks if requested
    if args.visualize:
        try:
            print("\nVisualizing attention mask...")
            fig = gasa_service.visualize_mask(
                mask=mask,
                format=MaskFormat.DENSE,
                mask_type=MaskType.ATTENTION,
                title=f"GASA Attention Mask (max_hops={args.max_hops})",
                show=True,
            )
            plt.savefig("gasa_attention_mask.png")
            print("Visualization saved to gasa_attention_mask.png")
        except Exception as e:
            print(f"Error visualizing mask: {e}")


def create_test_documents(num_documents: int) -> list[Document]:
    """Create test documents."""
    documents = []
    for i in range(num_documents):
        doc = Document(
            content=f"This is test document {i+1} with some content for testing GASA. "
            f"It contains information that relates to other documents in the set. "
            f"Specifically, it references concepts from documents {max(1, i-1)} "
            f"and {min(num_documents, i+2)}.",
            metadata=DocumentMetadata(
                source=f"test_doc_{i+1}.txt",
                document_id=f"doc_{i+1}",
            ),
        )
        documents.append(doc)
    return documents


def create_test_graph(documents: list[Document]) -> DependencyGraph:
    """Create a test dependency graph."""
    graph = DependencyGraph()

    # Add documents to graph
    nodes = []
    for doc in documents:
        node = graph.add_document_node(doc)
        nodes.append(node)

    # Add relationships
    for i in range(len(nodes)):
        # Connect to previous document
        if i > 0:
            graph.add_relationship(nodes[i].id, nodes[i - 1].id, "references")

        # Connect to next document
        if i < len(nodes) - 1:
            graph.add_relationship(nodes[i].id, nodes[i + 1].id, "references")

        # Add some long-range connections
        # Minimum index for long-range connections
        MIN_LONG_RANGE_INDEX = 2
        # Distance for long-range connections
        LONG_RANGE_DISTANCE = 3

        if i > MIN_LONG_RANGE_INDEX:
            graph.add_relationship(nodes[i].id, nodes[i - LONG_RANGE_DISTANCE].id, "references")

    return graph


if __name__ == "__main__":
    asyncio.run(main())
