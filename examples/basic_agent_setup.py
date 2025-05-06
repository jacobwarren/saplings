"""
This example demonstrates the most basic way to set up a Saplings agent
and run a simple query.
"""

from __future__ import annotations

import asyncio

from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore


async def main():
    # Create a memory store
    memory = MemoryStore()

    # Add a simple document
    await memory.add_document(
        "Saplings is a graph-first, self-improving agent framework that takes root in your repository or knowledge base, builds a structural map, and grows smarter each day."
    )

    # Create an agent with basic configuration
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
        )
    )

    # Set the memory store
    agent.memory_store = memory

    # Run a simple query
    result = await agent.run("What is Saplings?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
