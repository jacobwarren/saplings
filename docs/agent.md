# Agent

## Overview

The Agent class is the main entry point for using Saplings. It integrates all components of the framework, including:

- Memory and retrieval
- Planning and execution
- Validation and judging
- Self-healing and improvement
- Monitoring and tracing
- Graph-Aligned Sparse Attention (GASA)

## Agent Configuration

The Agent class is configured using the AgentConfig class:

```python
from saplings import Agent, AgentConfig

# Create agent configuration
config = AgentConfig(
    model_uri="openai://gpt-4",
    memory_path="./agent_memory",
    output_dir="./agent_output",
    enable_gasa=True,
    enable_monitoring=True,
    enable_self_healing=True,
    enable_tool_factory=True,
    max_tokens=2048,
    temperature=0.7,
    gasa_max_hops=2,
    retrieval_entropy_threshold=0.1,
    retrieval_max_documents=10,
    planner_budget_strategy="token_count",
    executor_verification_strategy="judge",
    tool_factory_sandbox_enabled=True,
    allowed_imports=["numpy", "pandas", "matplotlib"],
)

# Create agent
agent = Agent(config=config)
```

## Key Parameters

- `model_uri`: URI of the model to use (e.g., "openai://gpt-4", "anthropic://claude-3-opus")
- `memory_path`: Path to store agent memory
- `output_dir`: Directory to save outputs
- `enable_gasa`: Whether to enable Graph-Aligned Sparse Attention
- `enable_monitoring`: Whether to enable monitoring and tracing
- `enable_self_healing`: Whether to enable self-healing capabilities
- `enable_tool_factory`: Whether to enable dynamic tool creation
- `max_tokens`: Maximum number of tokens for model responses
- `temperature`: Temperature for model generation
- `gasa_max_hops`: Maximum number of hops for GASA mask
- `retrieval_entropy_threshold`: Entropy threshold for retrieval termination
- `retrieval_max_documents`: Maximum number of documents to retrieve
- `planner_budget_strategy`: Strategy for budget allocation
- `executor_verification_strategy`: Strategy for output verification
- `tool_factory_sandbox_enabled`: Whether to enable sandbox for tool execution
- `allowed_imports`: List of allowed imports for dynamic tools

## Basic Usage

```python
from saplings import Agent, AgentConfig
import asyncio

# Create agent configuration
config = AgentConfig(
    model_uri="openai://gpt-4",
    memory_path="./agent_memory",
)

# Create agent
agent = Agent(config=config)

# Run a task
async def main():
    result = await agent.run("Explain the concept of self-improving AI")
    print(result["final_result"])

# Run the async function
asyncio.run(main())
```

## Adding Documents to Memory

```python
from saplings import Agent, AgentConfig
from saplings.memory import Document, DocumentMetadata
import asyncio

# Create agent
agent = Agent(AgentConfig(model_uri="openai://gpt-4"))

# Add a document
async def main():
    document = Document(
        content="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
        metadata=DocumentMetadata(
            source="textbook",
            author="John Doe",
            date="2023-01-01"
        ),
    )
    
    # Add to memory store
    agent.memory_store.add_document(document)
    
    # Run a task that uses the document
    result = await agent.run("Explain machine learning")
    print(result["final_result"])

# Run the async function
asyncio.run(main())
```

## Retrieving Documents

```python
from saplings import Agent, AgentConfig
import asyncio

# Create agent
agent = Agent(AgentConfig(model_uri="openai://gpt-4"))

# Retrieve documents
async def main():
    # Retrieve documents related to a query
    documents = await agent.retrieve("machine learning")
    
    # Print retrieved documents
    for doc in documents:
        print(f"Document: {doc.id}")
        print(f"Content: {doc.content}")
        print(f"Metadata: {doc.metadata}")
        print()

# Run the async function
asyncio.run(main())
```

## Planning and Execution

```python
from saplings import Agent, AgentConfig
import asyncio

# Create agent
agent = Agent(AgentConfig(model_uri="openai://gpt-4"))

# Plan and execute
async def main():
    # Create a plan
    plan = await agent.plan("Analyze the performance of a sorting algorithm")
    
    # Print plan steps
    for i, step in enumerate(plan):
        print(f"Step {i+1}: {step.description}")
    
    # Execute the plan
    results = await agent.execute_plan(plan)
    
    # Print results
    for result in results["results"]:
        print(f"Step: {result['step'].description}")
        print(f"Result: {result['result']}")
        print()

# Run the async function
asyncio.run(main())
```

## Using GASA

```python
from saplings import Agent, AgentConfig
import asyncio

# Create agent with GASA enabled
config = AgentConfig(
    model_uri="openai://gpt-4",
    enable_gasa=True,
    gasa_max_hops=2,
)

agent = Agent(config=config)

# Run a task with GASA
async def main():
    result = await agent.run("Analyze the architecture of this codebase")
    print(result["final_result"])

# Run the async function
asyncio.run(main())
```

## Monitoring and Tracing

```python
from saplings import Agent, AgentConfig
from saplings.monitoring import TraceViewer
import asyncio

# Create agent with monitoring enabled
config = AgentConfig(
    model_uri="openai://gpt-4",
    enable_monitoring=True,
)

agent = Agent(config=config)

# Run a task with tracing
async def main():
    result = await agent.run("Explain the concept of self-improving AI")
    
    # Get the trace ID
    trace_id = result["trace_id"]
    
    # View the trace
    viewer = TraceViewer(trace_manager=agent.trace_manager)
    viewer.view_trace(
        trace_id=trace_id,
        output_path="trace.html",
        show=True,
    )

# Run the async function
asyncio.run(main())
```

## Self-Healing

```python
from saplings import Agent, AgentConfig
import asyncio

# Create agent with self-healing enabled
config = AgentConfig(
    model_uri="openai://gpt-4",
    enable_self_healing=True,
)

agent = Agent(config=config)

# Run a task and collect success pairs
async def main():
    # Run a task
    result = await agent.run("Implement a sorting algorithm")
    
    # Collect success pairs
    success_pair = {
        "input": "Implement a sorting algorithm",
        "output": result["final_result"],
        "score": result["judgment"]["score"],
    }
    
    # Add to success pair collector
    agent.success_pair_collector.add_pair(success_pair)
    
    # Generate a patch based on collected success pairs
    patch = await agent.patch_generator.generate_patch(
        code=result["final_result"],
        error="The sorting algorithm has O(n^2) time complexity",
    )
    
    print(f"Generated patch: {patch}")

# Run the async function
asyncio.run(main())
```

## Tool Factory

```python
from saplings import Agent, AgentConfig
from saplings.tool_factory import ToolSpecification
import asyncio

# Create agent with tool factory enabled
config = AgentConfig(
    model_uri="openai://gpt-4",
    enable_tool_factory=True,
)

agent = Agent(config=config)

# Create and use a tool
async def main():
    # Create a tool specification
    tool_spec = ToolSpecification(
        id="data_visualizer",
        name="Data Visualizer",
        description="Creates visualizations from data",
        template_id="python_tool",
        parameters={
            "function_name": "visualize_data",
            "parameters": "data: dict, output_path: str",
            "description": "Creates a bar chart visualization from data",
            "code_body": """
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
df.plot(kind='bar')
plt.savefig(output_path)
return output_path
"""
        }
    )
    
    # Create the tool
    tool = await agent.tool_factory.create_tool(tool_spec)
    
    # Use the tool
    result = tool().execute(
        data=[{"x": 1, "y": 10}, {"x": 2, "y": 20}],
        output_path="visualization.png"
    )
    
    print(f"Tool result: {result}")

# Run the async function
asyncio.run(main())
```

## Advanced Configuration

```python
from saplings import Agent, AgentConfig
from saplings.memory.config import MemoryConfig, VectorStoreConfig, GraphConfig
from saplings.gasa import GASAConfig
from saplings.executor import ExecutorConfig
from saplings.planner import PlannerConfig
from saplings.retrieval import RetrievalConfig
from saplings.monitoring import MonitoringConfig
import asyncio

# Create detailed configurations
memory_config = MemoryConfig(
    vector_store=VectorStoreConfig(
        store_type="in_memory",
        dimension=768,
    ),
    graph=GraphConfig(
        enable_graph=True,
        max_entities_per_document=50,
    ),
)

gasa_config = GASAConfig(
    max_hops=2,
    mask_strategy="binary",
    cache_masks=True,
)

executor_config = ExecutorConfig(
    enable_gasa=True,
    max_tokens=2048,
    temperature=0.7,
    verification_strategy="judge",
)

planner_config = PlannerConfig(
    budget_strategy="token_count",
    max_steps=10,
    cost_heuristic="linear",
)

retrieval_config = RetrievalConfig(
    entropy_threshold=0.1,
    max_documents=10,
    enable_graph_expansion=True,
)

monitoring_config = MonitoringConfig(
    enable_tracing=True,
    visualization_output_dir="./visualizations",
)

# Create agent configuration
config = AgentConfig(
    model_uri="openai://gpt-4",
    memory_config=memory_config,
    gasa_config=gasa_config,
    executor_config=executor_config,
    planner_config=planner_config,
    retrieval_config=retrieval_config,
    monitoring_config=monitoring_config,
)

# Create agent
agent = Agent(config=config)

# Run a task
async def main():
    result = await agent.run("Analyze the architecture of this codebase")
    print(result["final_result"])

# Run the async function
asyncio.run(main())
```

## Performance Considerations

- **Memory Usage**: The agent can consume significant memory when working with large document sets.
- **Model Costs**: Using powerful models like GPT-4 or Claude 3 Opus can incur significant API costs.
- **Latency**: Complex tasks with multiple steps can take time to complete.

## Best Practices

- **Start Simple**: Begin with basic configurations and add complexity as needed.
- **Use Appropriate Models**: Match the model to the task complexity.
- **Monitor Performance**: Use the monitoring tools to track performance and identify bottlenecks.
- **Tune Parameters**: Adjust parameters like `max_tokens`, `temperature`, and `gasa_max_hops` to optimize performance.
- **Leverage Memory**: Pre-populate the memory store with relevant documents to improve retrieval quality.
