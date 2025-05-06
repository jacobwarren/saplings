"""
This example shows how to index and query different types of documents,
demonstrating Saplings' ability to handle various content types.
"""

from __future__ import annotations

import asyncio

from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore


async def main():
    # Create a memory store
    memory = MemoryStore()

    # Add documents of different types
    await memory.add_document(
        "This is a plain text document about AI frameworks.",
        metadata={"type": "text", "topic": "ai"},
    )

    # Add some code
    python_code = """
    def hello_world():
        print("Hello from Saplings!")
        return "Success"
    """
    await memory.add_document(python_code, metadata={"type": "code", "language": "python"})

    # Create an agent
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
        )
    )

    # Set the memory store
    agent.memory_store = memory

    # Query that requires understanding different document types
    result = await agent.run("What types of documents have been added and what do they contain?")
    print(result)

    # Query specifically about the code
    result = await agent.run("What does the Python function do?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
