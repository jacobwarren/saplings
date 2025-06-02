# Saplings Public API

This module provides the stable, public API for the Saplings framework. All public interfaces should be imported from this module.

## API Structure

The API is organized into the following categories:

- **Agent**: Core agent functionality
- **Models**: Model adapters and LLM interfaces
- **Tools**: Tool definitions and utilities
- **Memory**: Memory store and dependency graph
- **Services**: Service builders and interfaces

## Usage

To use the Saplings API, import components directly from the top-level `saplings` package:

```python
from saplings import Agent, AgentConfig, PythonInterpreterTool

# Create an agent
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        tools=[PythonInterpreterTool()],
    )
)

# Run a task
import asyncio
result = asyncio.run(agent.run("Explain what Saplings is"))
print(result)
```

## Builder Pattern

For more complex configurations, use the builder pattern:

```python
from saplings import AgentBuilder

# Create an agent using the builder pattern
agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_memory_path("./agent_memory") \
    .with_output_dir("./agent_output") \
    .with_gasa_enabled(True) \
    .with_monitoring_enabled(True) \
    .with_model_parameters({
        "temperature": 0.7,
        "max_tokens": 2048,
    }) \
    .build()
```

## API Stability

The components exposed through this API are considered stable and will follow semantic versioning for any changes. Internal components not exposed through this API may change without notice.

## Internal vs. Public APIs

- **Public API**: Components exposed through the `saplings` package
- **Internal API**: Components that must be imported from specific submodules (e.g., `saplings.core.interfaces`)

Always prefer using the public API for application code. The internal API is subject to change without notice.
