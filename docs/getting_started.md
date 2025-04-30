# Getting Started with Saplings

This guide will help you get started with Saplings, a graphs-first, self-improving agent framework.

## Installation

```bash
pip install saplings
```

For optional features:

```bash
# Install with LoRA fine-tuning support
pip install saplings[lora]

# Install with all optional dependencies
pip install saplings[all]
```

## Quick Start

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore
import asyncio

# Initialize a memory store with your repository
memory = MemoryStore()
memory.index_repository("/path/to/your/repo")

# Create an agent configuration
config = AgentConfig(
    model_uri="openai://gpt-4",
    memory_path="./agent_memory",
)

# Create an agent
agent = Agent(config=config)

# Set the memory store
agent.memory_store = memory

# Run a task (note: this is an async method)
async def main():
    result = await agent.run("Explain the architecture of this codebase")
    print(result["final_result"])

# Run the async function
asyncio.run(main())
```

## Basic Concepts

Saplings is built around several core components:

1. **Memory**: Stores and indexes your knowledge base
2. **Retrieval**: Finds relevant information from memory
3. **Planning**: Creates a plan to accomplish a task
4. **Execution**: Executes the plan and generates output
5. **JudgeAgent & Validator**: Evaluates and improves the output
6. **Self-Healing**: Automatically fixes errors and improves over time

## Next Steps

- Learn about the [Core Concepts](./core_concepts.md)
- Explore the [Memory System](./memory.md)
- Understand [Self-Healing and Adaptation](./self_healing.md)
- Check out the [Examples](./examples.md)
