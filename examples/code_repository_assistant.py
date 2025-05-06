"""
This example shows how to build a code repository assistant that can
analyze, explain, and suggest improvements to an existing codebase.
"""

from __future__ import annotations

import asyncio
import os

from saplings import Agent, AgentConfig
from saplings.memory import DependencyGraph, MemoryStore


async def main():
    # Create memory components
    memory = MemoryStore()
    dependency_graph = DependencyGraph()

    # Index a repository (path to your project)
    # Replace with your own repository path
    repo_path = os.path.abspath("../")  # Default to parent directory
    print(f"Indexing repository at: {repo_path}")
    memory.index_repository(repo_path)

    # Build dependency graph based on imports, function calls, etc.
    dependency_graph.build_from_memory(memory)

    # Create agent with GASA enabled
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,  # Enable Graph-Aligned Sparse Attention
            gasa_max_hops=2,  # Maximum graph hops for attention
            memory_path="./repo_assistant_memory",  # Persist memory
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = dependency_graph

    # Example tasks
    tasks = [
        "Explain the architecture of this codebase",
        "What are the main modules and how do they interact?",
        "Identify any potential memory leaks or performance bottlenecks",
        "Suggest improvements to the error handling",
    ]

    # Run each task
    for task in tasks:
        print(f"\n--- Task: {task} ---\n")
        result = await agent.run(task)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
