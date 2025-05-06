"""
This example demonstrates how to visualize Graph-Aligned Sparse Attention (GASA) masks
and analyze their impact on model performance and efficiency.
"""

from __future__ import annotations

import asyncio
import os

from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.gasa import GASAConfig, MaskBuilder, MaskFormat, MaskType, MaskVisualizer
from saplings.memory import DependencyGraph, MemoryStore


async def main():
    # Import necessary modules

    # Create model
    print("Creating model...")
    model = LLM.create("openai", "gpt-4o")

    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Add documents
    print("Adding documents to memory...")
    documents = [
        "Graph Attention Networks (GATs) are a type of neural network designed to operate on graph-structured data.",
        "Unlike standard neural networks, GATs can process data with arbitrary graph structure.",
        "The key innovation in GATs is the attention mechanism that allows nodes to attend to their neighbors' features.",
        "This attention mechanism computes weights between connected nodes, determining how much influence each neighbor has.",
        "GATs have been applied to various tasks including node classification, link prediction, and graph classification.",
    ]

    for i, doc in enumerate(documents):
        await memory.add_document(content=doc, metadata={"id": f"doc_{i}"})

    # Build dependency graph with custom relationships
    print("Building dependency graph...")
    await graph.build_from_memory(memory)

    # Add specific relationships
    graph.add_relationship("doc_0", "doc_1", "relates_to", 0.9)
    graph.add_relationship("doc_0", "doc_2", "relates_to", 0.8)
    graph.add_relationship("doc_2", "doc_3", "builds_on", 1.0)
    graph.add_relationship("doc_2", "doc_4", "exemplifies", 0.7)
    graph.add_relationship("doc_3", "doc_4", "relates_to", 0.6)

    # Configure GASA
    print("Configuring GASA...")
    gasa_config = GASAConfig(
        max_hops=2,
        mask_strategy="binary",
        visualize=True,
        visualization_dir="./gasa_visualizations",
    )

    # Create visualization directory if it doesn't exist
    if not os.path.exists(gasa_config.visualization_dir):
        os.makedirs(gasa_config.visualization_dir)

    # Create mask builder with tokenizer
    tokenizer = model.get_tokenizer()
    mask_builder = MaskBuilder(
        graph=graph,
        config=gasa_config,
        tokenizer=tokenizer,
    )

    # Create visualizer
    visualizer = MaskVisualizer()

    # Create prompt
    prompt = """
    Analyze the following information about Graph Attention Networks:
    1. What are the key innovations in GATs?
    2. How do they differ from standard neural networks?
    3. What are their applications?
    """

    # Build mask
    print("Building attention mask...")
    mask = mask_builder.build_mask(
        documents=memory.get_documents(),
        prompt=prompt,
        format=MaskFormat.DENSE,
        mask_type=MaskType.ATTENTION,
    )

    # Visualize mask
    print("Visualizing mask...")
    visualization_path = os.path.join(gasa_config.visualization_dir, "gasa_attention_mask.png")
    visualizer.visualize_mask(
        mask=mask,
        format=MaskFormat.DENSE,
        mask_type=MaskType.ATTENTION,
        output_path=visualization_path,
        title="GASA Attention Mask",
        show=False,
    )
    print(f"Saved visualization to {visualization_path}")

    # Analyze mask statistics
    print("Analyzing mask statistics...")
    stats = visualizer.analyze_mask_statistics(
        mask=mask,
        format=MaskFormat.DENSE,
        mask_type=MaskType.ATTENTION,
    )

    # Print stats
    print(f"Mask shape: {stats.get('shape', 'N/A')}")
    print(f"Sparsity: {stats.get('sparsity', 0):.2%}")
    print(f"Total FLOPs: {stats.get('flops', 0):,}")
    print(f"FLOPs reduction: {stats.get('flops_reduction', 0):.2%}")

    # Create agent with GASA
    print("Creating agent with GASA...")
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,
            gasa_strategy=gasa_config.mask_strategy,
            gasa_max_hops=gasa_config.max_hops,
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Execute with GASA
    print("\nExecuting with GASA enabled...")
    with_gasa_result = await agent.run(prompt)

    # Execute without GASA
    print("\nExecuting without GASA...")
    agent.config.enable_gasa = False
    without_gasa_result = await agent.run(prompt)

    # Compare results
    print("\n=== Results with GASA ===")
    print(with_gasa_result)

    print("\n=== Results without GASA ===")
    print(without_gasa_result)


if __name__ == "__main__":
    asyncio.run(main())
