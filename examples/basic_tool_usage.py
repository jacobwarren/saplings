"""
This example demonstrates how to use built-in tools with a Saplings agent.
"""

from __future__ import annotations

import asyncio

from saplings import Agent, AgentConfig
from saplings.tools import PythonInterpreterTool, WikipediaSearchTool


async def main():
    # Create tools
    python_tool = PythonInterpreterTool()
    wiki_tool = WikipediaSearchTool()

    # Create an agent with tools
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=[python_tool, wiki_tool],
        )
    )

    # Run a task that requires using tools
    result = await agent.run(
        "Search for information about Graph Attention Networks on Wikipedia, "
        "then write a Python function that creates a simple representation of "
        "a graph attention mechanism."
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
