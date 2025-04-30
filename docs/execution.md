# Execution

## Overview

The Execution system in Saplings carries out tasks using language models and tools. It supports features like speculative execution, verification, and Graph-Aligned Sparse Attention (GASA).

## Key Components

The Execution system consists of several key components:

1. **Executor**: Executes tasks using language models and tools
2. **ExecutorConfig**: Configuration for the executor
3. **SpeculativeExecutor**: Performs speculative execution for improved efficiency
4. **VerificationFlow**: Verifies the output of execution

## Basic Usage

```python
from saplings.executor import Executor, ExecutorConfig
from saplings.core.model_adapter import LLM
import asyncio

# Create a model
model = LLM.from_uri("openai://gpt-4")

# Create an executor configuration
config = ExecutorConfig(
    max_tokens=2048,
    temperature=0.7,
    verification_strategy="judge",
)

# Create an executor
executor = Executor(
    model=model,
    config=config,
)

# Execute a task
async def main():
    result = await executor.execute(
        prompt="Explain the concept of self-improving AI",
    )
    
    print(f"Result: {result.text}")
    print(f"Token usage: {result.usage}")

# Run the async function
asyncio.run(main())
```

## Integration with Agent

The Execution system is integrated with the Agent class:

```python
from saplings import Agent, AgentConfig
import asyncio

# Create agent
agent = Agent(AgentConfig(model_uri="openai://gpt-4"))

# Execute a task
async def main():
    result = await agent.execute(
        prompt="Explain the concept of self-improving AI",
    )
    
    print(f"Result: {result['text']}")

# Run the async function
asyncio.run(main())
```

## Advanced Features

### Speculative Execution

The SpeculativeExecutor performs speculative execution for improved efficiency:

```python
from saplings.executor import SpeculativeExecutor, ExecutorConfig
from saplings.core.model_adapter import LLM
import asyncio

# Create models
draft_model = LLM.from_uri("openai://gpt-3.5-turbo")
verify_model = LLM.from_uri("openai://gpt-4")

# Create an executor configuration
config = ExecutorConfig(
    max_tokens=2048,
    temperature=0.7,
    enable_speculative=True,
)

# Create a speculative executor
executor = SpeculativeExecutor(
    draft_model=draft_model,
    verify_model=verify_model,
    config=config,
)

# Execute a task with speculative execution
async def main():
    result = await executor.execute(
        prompt="Explain the concept of self-improving AI",
    )
    
    print(f"Result: {result.text}")
    print(f"Token usage: {result.usage}")
    print(f"Speculative efficiency: {result.metadata.get('speculative_efficiency', 0)}")

# Run the async function
asyncio.run(main())
```

### Verification Flow

The VerificationFlow verifies the output of execution:

```python
from saplings.executor import Executor, ExecutorConfig, VerificationFlow
from saplings.core.model_adapter import LLM
import asyncio

# Create models
model = LLM.from_uri("openai://gpt-4")
verify_model = LLM.from_uri("openai://gpt-4")

# Create an executor configuration
config = ExecutorConfig(
    max_tokens=2048,
    temperature=0.7,
    verification_strategy="model",
)

# Create an executor
executor = Executor(
    model=model,
    config=config,
)

# Create a verification flow
verification_flow = VerificationFlow(
    model=verify_model,
    strategy="model",
)

# Execute a task with verification
async def main():
    result = await executor.execute(
        prompt="Explain the concept of self-improving AI",
    )
    
    # Verify the result
    verification_result = await verification_flow.verify(
        prompt="Explain the concept of self-improving AI",
        output=result.text,
    )
    
    print(f"Result: {result.text}")
    print(f"Verification result: {verification_result.is_valid}")
    print(f"Verification feedback: {verification_result.feedback}")

# Run the async function
asyncio.run(main())
```

### Graph-Aligned Sparse Attention (GASA)

The Executor supports Graph-Aligned Sparse Attention (GASA) for improved efficiency and context-awareness:

```python
from saplings.executor import Executor, ExecutorConfig
from saplings.gasa import GASAConfig
from saplings.core.model_adapter import LLM
from saplings.memory import DependencyGraph
import asyncio

# Create a model
model = LLM.from_uri("openai://gpt-4")

# Create a dependency graph
graph = DependencyGraph()

# Add nodes and edges to the graph
# ...

# Create a GASA configuration
gasa_config = GASAConfig(
    max_hops=2,
    mask_strategy="binary",
    cache_masks=True,
)

# Create an executor configuration
executor_config = ExecutorConfig(
    enable_gasa=True,
    max_tokens=2048,
    temperature=0.7,
)

# Create an executor with GASA
executor = Executor(
    model=model,
    config=executor_config,
    gasa_config=gasa_config,
    dependency_graph=graph,
)

# Execute a task with GASA
async def main():
    documents = [...]  # List of documents
    result = await executor.execute(
        prompt="Summarize these documents:",
        documents=documents,
    )
    
    print(f"Result: {result.text}")
    print(f"Token usage: {result.usage}")
    print(f"GASA stats: {result.metadata.get('gasa_stats', {})}")

# Run the async function
asyncio.run(main())
```

## Performance Considerations

- **Speculative Execution**: Speculative execution can significantly improve efficiency but may reduce quality
- **Verification**: Verification improves quality but increases cost
- **GASA**: GASA improves efficiency and context-awareness but adds complexity
- **Model Quality**: The quality of the model used for execution affects the quality of the output

## Best Practices

- **Balance Efficiency and Quality**: Choose the right combination of speculative execution and verification
- **Use GASA When Appropriate**: Enable GASA when working with structured documents and relationships
- **Monitor Execution Performance**: Track token usage and execution time to optimize parameters
- **Use the Right Model for the Task**: Match the model to the complexity of the task
