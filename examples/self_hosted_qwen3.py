"""
This example demonstrates using Saplings with a self-hosted Qwen3 model via vLLM.
Self-hosting models gives you more control, privacy, and can reduce costs for high-volume applications.
"""

from __future__ import annotations

import asyncio

from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM
from saplings.memory import MemoryStore


async def main():
    # Create a memory store
    memory = MemoryStore()

    # Add some documents
    await memory.add_document(
        "Saplings is a graph-first, self-improving agent framework that takes root in your repository.",
        metadata={"type": "documentation", "section": "overview"},
    )

    await memory.add_document(
        "GASA (Graph-Aligned Sparse Attention) is a technique that improves efficiency by focusing attention on relevant context.",
        metadata={"type": "documentation", "section": "gasa"},
    )

    # Configure a self-hosted Qwen3 model using vLLM
    # Note: You must have vLLM installed and the model downloaded locally
    print("Creating vLLM model with Qwen/Qwen3-1.7B...")
    model = LLM.create(
        provider="vllm",
        model_name="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,  # Set based on number of GPUs available
        gpu_memory_utilization=0.8,  # Adjust based on your system
        quantization="awq",  # Optional: use quantization to reduce memory usage
    )

    # Create an agent with self-hosted model
    print("Creating agent with self-hosted model...")
    agent = Agent(
        config=AgentConfig(
            provider="vllm",
            model_name="Qwen/Qwen3-1.7B",
            # vLLM specific parameters
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            quantization="awq",
            # Enable GASA - fully supported with vLLM since we have direct access to attention
            enable_gasa=True,
            gasa_max_hops=2,
        )
    )

    # Set the memory store
    agent.memory_store = memory

    # Optionally, set the model directly if you want to share it between components
    # agent.model = model

    # Run a query
    print("\nRunning query about GASA...")
    result = await agent.run("Explain how GASA relates to the Saplings framework.")
    print(f"Result: {result}")

    # Run more complex tasks that leverage the local model's capabilities
    print("\nRunning complex task...")
    result = await agent.run("""
    Write a short Python function that implements a simple graph traversal algorithm
    to find the shortest path between two nodes.
    """)
    print(f"Result: {result}")

    # Multi-turn conversations are also supported
    print("\nStarting multi-turn conversation...")
    messages = [
        {"role": "user", "content": "What are the advantages of self-hosting models like Qwen3?"}
    ]
    result = await agent.model.chat(messages)
    print(f"Model response: {result.text}")

    # Add to the conversation
    messages.append({"role": "assistant", "content": result.text})
    messages.append({"role": "user", "content": "How does quantization affect model performance?"})

    result = await agent.model.chat(messages)
    print(f"Model response: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
