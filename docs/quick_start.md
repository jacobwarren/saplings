# Quick Start Guide

This guide will help you get started with Saplings quickly, showing you how to install the library, set up a basic agent, add memory, use tools, and run a simple task.

## Installation

### Prerequisites

- Python 3.9+
- Poetry (recommended for dependency management)

### Install with pip

```bash
pip install saplings
```

### Install with Poetry

```bash
# Clone the repository
git clone https://github.com/jacobwarren/saplings.git
cd saplings

# Install dependencies
poetry install
```

## Creating Your First Agent

Let's create a simple agent that can answer questions:

```python
import asyncio
from saplings import Agent, AgentConfig

async def main():
    # Create an agent with basic configuration
    agent = Agent(
        config=AgentConfig(
            provider="openai",  # Use OpenAI as the provider
            model_name="gpt-4o",  # Use GPT-4o model
        )
    )

    # Run a simple query
    result = await agent.run("What is the capital of France?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Adding Memory

Let's enhance our agent by adding memory:

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore

async def main():
    # Create a memory store
    memory = MemoryStore()

    # Add some documents to memory
    await memory.add_document(
        "Paris is the capital and most populous city of France. It has an estimated population of 2,165,423 residents in 2019 in an area of more than 105 km²."
    )

    await memory.add_document(
        "France is a country primarily located in Western Europe. It also includes overseas regions and territories in the Americas and the Atlantic, Pacific and Indian Oceans."
    )

    # Create an agent with the memory store
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
        )
    )

    # Set the memory store
    agent.memory_store = memory

    # Run a query that will use the memory
    result = await agent.run("What is the capital of France and what is its population?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Using Tools

Now let's add tools to our agent:

```python
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
            tools=[python_tool, wiki_tool],  # Add tools to the agent
        )
    )

    # Run a task that requires using tools
    result = await agent.run(
        "Search for information about the Eiffel Tower on Wikipedia, "
        "then write a Python function that calculates how long it would take "
        "an object to fall from the top of the Eiffel Tower, assuming no air resistance."
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Enabling GASA

Graph-Aligned Sparse Attention (GASA) is a key feature of Saplings. Let's enable it:

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph

async def main():
    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Add documents to memory
    await memory.add_document(
        "The Eiffel Tower is 330 meters tall.",
        metadata={"source": "facts.txt", "id": "doc1"}
    )

    await memory.add_document(
        "The formula for the time it takes an object to fall is t = sqrt(2h/g), where h is height and g is 9.8 m/s².",
        metadata={"source": "physics.txt", "id": "doc2"}
    )

    # Build dependency graph
    await graph.add_edge("doc1", "doc2", "related_to")
    memory.dependency_graph = graph

    # Create an agent with GASA enabled
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,  # Enable GASA
            gasa_max_hops=2,   # Maximum graph hops for attention
        )
    )

    # Set the memory store
    agent.memory_store = memory

    # Run a query that will benefit from GASA
    result = await agent.run("How long would it take an object to fall from the top of the Eiffel Tower?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Planning and Execution

For more complex tasks, you can use the planning system:

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.planner import PlannerConfig, BudgetStrategy

async def main():
    # Create an agent with custom planner configuration
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            planner_config=PlannerConfig(
                total_budget=1.0,
                budget_strategy=BudgetStrategy.PROPORTIONAL,
                max_steps=5,
            ),
        )
    )

    # Create a plan
    plan = await agent.plan(
        task="Research the history of the Eiffel Tower, identify key facts, and create a timeline."
    )

    # Print the plan
    print("Plan steps:")
    for step in plan:
        print(f"- {step.task_description}")

    # Execute the plan
    success, result = await agent.execute_plan(plan)

    # Print the result
    print("\nResult:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

Now that you've created your first Saplings agent, here are some next steps to explore:

1. **Explore the examples directory** for more advanced usage patterns
2. **Read the documentation** to understand the core concepts and components
3. **Try different model providers** like vLLM with Qwen models
4. **Experiment with multimodal agents** that can handle text, images, and more
5. **Implement self-healing** to make your agents more robust

For more detailed information, check out the following documentation:

- [Core Concepts](./core_concepts.md)
- [Memory and Retrieval](./memory.md)
- [GASA Implementation](./gasa.md)
- [Planning and Execution](./planner.md)
- [Tools System](./tools.md)
- [Examples](./examples.md)
