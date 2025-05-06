# Agent

The Agent class is the primary entry point for using Saplings, providing a clean, intuitive API that integrates all components of the framework through composition and delegation.

## Overview

The Agent class is designed to be the main interface for using Saplings, hiding the complexity of the underlying components while providing a powerful and flexible API. It follows key design principles:

- **Dependency Inversion**: Agent depends on interfaces, not concrete implementations
- **Composition over Inheritance**: Uses internal AgentFacade instead of inheritance
- **Single Responsibility**: Each component has a clear, focused responsibility
- **Interface Segregation**: Clear, cohesive interfaces for each service

The Agent class integrates all the key features of Saplings:

- **Structural Memory**: Vector and graph stores for rich knowledge representation
- **Cascaded Retrieval**: TF-IDF → embeddings → graph expansion for optimal context
- **Planning and Execution**: Task breakdown and controlled execution
- **Validation and Self-Improvement**: Output validation and continuous learning
- **Tool Integration**: Built-in and custom tools for enhanced capabilities
- **Multimodal Support**: Handling different modalities (text, image, audio, video)
- **Monitoring**: Comprehensive tracing and performance analysis

## Core Concepts

### Agent Configuration

The `AgentConfig` class centralizes all configuration options for the Agent, making it easy to customize behavior while providing sensible defaults:

- **Model Settings**: Provider, model name, and parameters
- **Memory Settings**: Memory path and configuration
- **GASA Settings**: Graph-Aligned Sparse Attention configuration
- **Retrieval Settings**: Entropy threshold and document limits
- **Planning Settings**: Budget strategy and constraints
- **Execution Settings**: Verification strategy
- **Tool Settings**: Tool factory configuration and allowed imports
- **Modality Settings**: Supported input and output modalities

### Agent Lifecycle

The typical lifecycle of an Agent involves:

1. **Initialization**: Creating an Agent with a configuration
2. **Memory Population**: Adding documents to the agent's memory
3. **Tool Registration**: Registering tools for enhanced capabilities
4. **Task Execution**: Running tasks with the agent
5. **Self-Improvement**: Learning from past performance

### Agent Facade

Internally, the Agent uses an AgentFacade that coordinates between various services:

- **ModelService**: Manages the LLM
- **MemoryManager**: Manages memory storage and retrieval
- **RetrievalService**: Handles document retrieval
- **PlannerService**: Creates plans for tasks
- **ExecutionService**: Executes prompts with context
- **ValidatorService**: Validates outputs
- **SelfHealingService**: Handles error recovery and learning
- **ToolService**: Manages tools
- **ModalityService**: Handles different modalities
- **MonitoringService**: Provides tracing and monitoring
- **OrchestrationService**: Coordinates multi-agent workflows

## API Reference

### Agent

```python
class Agent:
    def __init__(self, config: AgentConfig):
        """
        Initialize the agent with the provided configuration.

        Args:
            config: Agent configuration
        """

    async def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a document to the agent's memory.

        Args:
            content: Document content
            metadata: Optional metadata for the document

        Returns:
            Document: The added document
        """

    async def execute_plan(self, plan, context=None, use_tools=True):
        """
        Execute a plan.

        Args:
            plan: Plan to execute
            context: Optional context for execution
            use_tools: Whether to use tools during execution

        Returns:
            Dict[str, Any]: Execution results
        """

    def register_tool(self, tool):
        """
        Register a tool with the agent.

        Args:
            tool: Tool to register

        Returns:
            bool: Whether the registration was successful
        """

    async def create_tool(self, name, description, code):
        """
        Create a dynamic tool.

        Args:
            name: Tool name
            description: Tool description
            code: Tool implementation code

        Returns:
            Tool: The created tool
        """

    async def judge_output(self, input_data, output_data, judgment_type="general"):
        """
        Judge an output using the JudgeAgent.

        Args:
            input_data: Input data
            output_data: Output data to judge
            judgment_type: Type of judgment to perform

        Returns:
            Dict[str, Any]: Judgment results
        """

    async def self_improve(self):
        """
        Improve the agent based on past performance.

        Returns:
            Dict[str, Any]: Self-improvement results
        """

    async def run(self, task, input_modalities=None, output_modalities=None, use_tools=True):
        """
        Run the agent on a task, handling the full lifecycle.

        Args:
            task: Task description
            input_modalities: Modalities of the input (default: ["text"])
            output_modalities: Expected modalities of the output (default: ["text"])
            use_tools: Whether to enable tool usage (default: True)

        Returns:
            Dict[str, Any]: Task results
        """

    async def add_documents_from_directory(self, directory, extension=".txt"):
        """
        Add documents from a directory.

        Args:
            directory: Directory path
            extension: File extension to filter by

        Returns:
            List[Document]: Added documents
        """

    async def retrieve(self, query, limit=None):
        """
        Retrieve documents based on a query.

        Args:
            query: Query string
            limit: Maximum number of documents to retrieve

        Returns:
            List[Tuple[Document, float]]: Retrieved documents with scores
        """

    async def plan(self, task, context=None):
        """
        Create a plan for a task.

        Args:
            task: Task description
            context: Optional context for planning

        Returns:
            Plan: The created plan
        """

    async def execute(self, prompt, context=None, use_tools=True):
        """
        Execute a prompt with the agent.

        Args:
            prompt: Prompt to execute
            context: Optional context for execution
            use_tools: Whether to use tools during execution

        Returns:
            Dict[str, Any]: Execution results
        """
```

### AgentConfig

```python
class AgentConfig:
    def __init__(
        self,
        provider: str,
        model_name: str,
        memory_path: str = "./agent_memory",
        output_dir: str = "./agent_output",
        enable_gasa: bool = True,
        enable_monitoring: bool = True,
        enable_self_healing: bool = True,
        self_healing_max_retries: int = 3,
        enable_tool_factory: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        gasa_max_hops: int = 2,
        gasa_strategy: str = "binary",
        gasa_fallback: str = "block_diagonal",
        gasa_shadow_model: bool = False,
        gasa_shadow_model_name: str = "Qwen/Qwen3-0.6B",
        gasa_prompt_composer: bool = False,
        retrieval_entropy_threshold: float = 0.1,
        retrieval_max_documents: int = 10,
        planner_budget_strategy: str = "token_count",
        planner_total_budget: float = 1.0,
        planner_allow_budget_overflow: bool = False,
        planner_budget_overflow_margin: float = 0.1,
        executor_verification_strategy: str = "judge",
        tool_factory_sandbox_enabled: bool = True,
        allowed_imports: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        supported_modalities: Optional[List[str]] = None,
        **model_parameters,
    ):
        """
        Initialize the agent configuration.

        Args:
            provider: Model provider (e.g., 'vllm', 'openai', 'anthropic')
            model_name: Model name
            memory_path: Path to store agent memory
            output_dir: Directory to save outputs
            enable_gasa: Whether to enable Graph-Aligned Sparse Attention
            enable_monitoring: Whether to enable monitoring and tracing
            enable_self_healing: Whether to enable self-healing capabilities
            self_healing_max_retries: Maximum number of retry attempts for self-healing operations
            enable_tool_factory: Whether to enable dynamic tool creation
            max_tokens: Maximum number of tokens for model responses
            temperature: Temperature for model generation
            gasa_max_hops: Maximum number of hops for GASA mask
            gasa_strategy: Strategy for GASA mask (binary, soft, learned)
            gasa_fallback: Fallback strategy for models that don't support sparse attention
            gasa_shadow_model: Whether to enable shadow model for tokenization
            gasa_shadow_model_name: Name of the shadow model to use
            gasa_prompt_composer: Whether to enable graph-aware prompt composer
            retrieval_entropy_threshold: Entropy threshold for retrieval termination
            retrieval_max_documents: Maximum number of documents to retrieve
            planner_budget_strategy: Strategy for budget allocation
            planner_total_budget: Total budget for the planner in USD
            planner_allow_budget_overflow: Whether to allow exceeding the budget
            planner_budget_overflow_margin: Margin by which the budget can be exceeded (as a fraction)
            executor_verification_strategy: Strategy for output verification
            tool_factory_sandbox_enabled: Whether to enable sandbox for tool execution
            allowed_imports: List of allowed imports for dynamic tools
            tools: List of tools to register with the agent
            supported_modalities: List of supported modalities (text, image, audio, video)
            **model_parameters: Additional model parameters
        """
```

## Usage Examples

### Basic Agent

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore

# Create a memory store
memory = MemoryStore()

# Add a document to memory
memory.add_document(
    content="Saplings is a graph-first, self-improving agent framework that takes root in your repository or knowledge base, builds a structural map, and grows smarter each day.",
    metadata={"source": "README.md"}
)

# Create an agent configuration
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    memory_path="./agent_memory"  # Directory where memory will be saved/loaded
)

# Create an agent with the configuration
agent = Agent(config=config)

# Set the memory store
agent.memory_store = memory

# Run a task
import asyncio
result = asyncio.run(agent.run("What is Saplings?"))
print(result)
```

### Agent with Tools

```python
from saplings import Agent, AgentConfig
from saplings.tools import PythonInterpreterTool, WikipediaSearchTool

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
import asyncio
result = asyncio.run(agent.run(
    "Search for information about Graph Attention Networks on Wikipedia, "
    "then write a Python function that creates a simple representation of "
    "a graph attention mechanism."
))
print(result)
```

### Agent with GASA

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph
from saplings.gasa import GASAConfig

# Initialize memory components
memory = MemoryStore()
graph = DependencyGraph()

# Add documents to memory
memory.add_document(
    content="Graph Attention Networks (GATs) are neural networks that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations.",
    metadata={"source": "paper.txt", "section": "introduction"}
)

# Create agent configuration with GASA enabled
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    enable_gasa=True,
    gasa_max_hops=2,
    gasa_strategy="binary",
    gasa_fallback="block_diagonal",
)

# Create agent with custom configuration
agent = Agent(config=config)

# Set memory components
agent.memory_store = memory
agent.dependency_graph = graph

# Run a task
import asyncio
result = asyncio.run(agent.run("Explain how Graph Attention Networks work"))
print(result)
```

### Multimodal Agent

```python
from saplings import Agent, AgentConfig
import asyncio

# Create agent configuration with multimodal support
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    memory_path="./agent_memory",
    supported_modalities=["text", "image"],  # Specify supported modalities
)

# Create agent
agent = Agent(config=config)

# Run a task with specific input and output modalities
result = asyncio.run(agent.run(
    task="Generate a description of a sunset over mountains and create an image of it.",
    input_modalities=["text"],
    output_modalities=["text", "image"]
))

# Process the result
text_output = result.get("text")
image_output = result.get("image")

print(text_output)
# Save the image if present
if image_output:
    with open("sunset.png", "wb") as f:
        f.write(image_output)
```

### Agent with Self-Healing

```python
from saplings import Agent, AgentConfig
from saplings.self_heal import PatchGenerator, SuccessPairCollector

# Create a patch generator
patch_generator = PatchGenerator(max_retries=3)

# Create a success pair collector
collector = SuccessPairCollector(output_dir="./success_pairs")

# Create an agent with self-healing enabled
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_self_healing=True,
        self_healing_max_retries=3,
    )
)

# Set the patch generator and success pair collector
agent.patch_generator = patch_generator
agent.success_pair_collector = collector

# Run a task
import asyncio
result = asyncio.run(agent.run(
    "Write a Python function to calculate the factorial of a number."
))

# The agent will automatically fix any errors in the generated code
print(result)
```

### Agent with Monitoring

```python
from saplings import Agent, AgentConfig
from saplings.monitoring import MonitoringConfig, TraceManager, TraceViewer

# Configure monitoring
monitoring_config = MonitoringConfig(
    visualization_output_dir="./visualizations",
)

# Create a trace manager
trace_manager = TraceManager(config=monitoring_config)

# Create an agent with monitoring enabled
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_monitoring=True,
    )
)

# Set the trace manager
agent.trace_manager = trace_manager

# Run a task
import asyncio
result = asyncio.run(agent.run("Explain the concept of graph-based memory"))

# Get the trace ID from the result
trace_id = result.get("trace_id")

# Visualize the trace
trace_viewer = TraceViewer(trace_manager=trace_manager)
trace_viewer.view_trace(
    trace_id=trace_id,
    output_path="agent_execution_trace.html",
    show=True,
)
```

### Agent with Custom Model

```python
from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM

# Create a custom model
model = LLM.create(
    provider="vllm",
    model_name="Qwen/Qwen3-7B-Instruct",
    device="cuda"
)

# Create an agent with the custom model
agent = Agent(
    config=AgentConfig(
        provider="vllm",
        model_name="Qwen/Qwen3-7B-Instruct",
        device="cuda",
    )
)

# Run a task
import asyncio
result = asyncio.run(agent.run("Explain the concept of graph-based memory"))
print(result)
```

## Advanced Features

### Dynamic Tool Creation

```python
from saplings import Agent, AgentConfig
from saplings.tools import ToolFactory

# Create an agent with tool factory enabled
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_tool_factory=True,
        tool_factory_sandbox_enabled=True,
        allowed_imports=["os", "json", "math", "numpy", "pandas", "matplotlib"],
    )
)

# Create a dynamic tool
import asyncio
visualization_tool = asyncio.run(agent.create_tool(
    name="VisualizationTool",
    description="Creates visualizations from data",
    code="""
import matplotlib.pyplot as plt
import numpy as np

def execute(data, output_path="visualization.png"):
    """Create a visualization from data."""
    # Convert data to numpy arrays
    x = [item["x"] for item in data]
    y = [item["y"] for item in data]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.title("Data Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # Save the plot
    plt.savefig(output_path)

    return {"status": "success", "output_path": output_path}
"""
))

# Use the tool
result = visualization_tool.execute(
    data=[{"x": 1, "y": 10}, {"x": 2, "y": 20}, {"x": 3, "y": 15}, {"x": 4, "y": 25}],
    output_path="visualization.png"
)
print(result)
```

### Judging Outputs

```python
from saplings import Agent, AgentConfig
from saplings.judge import JudgeAgent, JudgeConfig, Rubric, RubricItem, ScoringDimension

# Create a custom rubric
custom_rubric = Rubric(
    name="Code Quality Rubric",
    description="Rubric for evaluating code quality",
    items=[
        RubricItem(
            dimension=ScoringDimension.CORRECTNESS,
            weight=2.0,
            description="How correct and functional the code is",
            criteria={
                "0.0": "Code does not run or has major errors",
                "0.5": "Code runs but has minor issues",
                "1.0": "Code runs perfectly and handles all cases",
            },
        ),
        RubricItem(
            dimension=ScoringDimension.COHERENCE,
            weight=1.5,
            description="How well-structured and readable the code is",
            criteria={
                "0.0": "Code is poorly structured and hard to read",
                "0.5": "Code is somewhat structured but could be improved",
                "1.0": "Code is well-structured and easy to read",
            },
        ),
        RubricItem(
            dimension="efficiency",
            weight=1.0,
            description="How efficient the code is",
            criteria={
                "0.0": "Code is very inefficient",
                "0.5": "Code has average efficiency",
                "1.0": "Code is highly optimized and efficient",
            },
        ),
    ],
)

# Create an agent with a judge
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
    )
)

# Generate some code
import asyncio
code_result = asyncio.run(agent.run(
    "Write a Python function to find the nth Fibonacci number using dynamic programming."
))

# Judge the output
judgment = asyncio.run(agent.judge_output(
    input_data="Write a Python function to find the nth Fibonacci number using dynamic programming.",
    output_data=code_result,
    judgment_type=custom_rubric,
))

# Print the judgment
print(f"Overall score: {judgment['overall_score']}")
print(f"Passed: {judgment['passed']}")
print(f"Critique: {judgment['critique']}")
print("Dimension scores:")
for score in judgment["dimension_scores"]:
    print(f"  {score['dimension']}: {score['score']} - {score['explanation']}")
```

### Planning and Execution

```python
from saplings import Agent, AgentConfig

# Create an agent
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        planner_budget_strategy="token_count",
        planner_total_budget=1.0,
    )
)

# Create a plan
import asyncio
plan = asyncio.run(agent.plan(
    task="Analyze a dataset, create visualizations, and write a report."
))

# Print the plan
print("Plan:")
for i, step in enumerate(plan.steps):
    print(f"Step {i+1}: {step.description}")
    print(f"  Budget: {step.budget}")

# Execute the plan
result = asyncio.run(agent.execute_plan(plan))

# Print the result
print("\nExecution result:")
for step_id, step_result in result.items():
    print(f"Step {step_id}:")
    print(f"  Output: {step_result['output'][:100]}...")
```

### Memory Management

```python
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, Document, DocumentMetadata

# Create a memory store
memory = MemoryStore()

# Add documents with metadata
doc1 = memory.add_document(
    content="Graph Attention Networks (GATs) are neural networks that operate on graph-structured data.",
    metadata=DocumentMetadata(
        source="paper.txt",
        author="Veličković et al.",
        tags=["graph", "attention", "neural networks"],
    )
)

doc2 = memory.add_document(
    content="Graph-Aligned Sparse Attention (GASA) injects learned binary attention masks derived from retrieval dependency graphs into transformer layers.",
    metadata=DocumentMetadata(
        source="saplings_docs.txt",
        author="Saplings Team",
        tags=["gasa", "attention", "graph"],
    )
)

# Create an agent with the memory store
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        memory_path="./agent_memory",
    )
)
agent.memory_store = memory

# Save the memory store
memory.save("./agent_memory")

# Retrieve documents
import asyncio
results = asyncio.run(agent.retrieve(query="graph attention", limit=5))

# Print the results
print("Retrieved documents:")
for doc, score in results:
    print(f"Document: {doc.id}, Score: {score}")
    print(f"Content: {doc.content[:100]}...")
    print(f"Source: {doc.metadata.source}")
    print(f"Tags: {doc.metadata.tags}")
    print()
```

## Implementation Details

### Agent Implementation

The `Agent` class is implemented using composition rather than inheritance, delegating to an internal `AgentFacade` instance. This embraces the principle of "composition over inheritance" and allows for better flexibility and maintainability.

The Agent follows the Dependency Inversion Principle by depending on service interfaces rather than concrete implementations, allowing for easier testing, extension, and alternative implementations.

### Run Method Implementation

The `run` method orchestrates the entire agent workflow:

1. **Retrieve Context**: Retrieve relevant documents from memory
2. **Create Plan**: Break down the task into steps
3. **Execute Plan**: Execute each step with the appropriate context
4. **Validate Results**: Validate and judge the results
5. **Self-Improvement**: Collect success pairs for self-improvement

### Tool Integration

Tools are integrated into the agent through the `ToolService`, which manages tool registration, discovery, and execution. The agent can use both built-in tools and custom tools created dynamically through the `ToolFactory`.

### Memory Integration

Memory is integrated through the `MemoryManager`, which manages the `MemoryStore` and `DependencyGraph`. The agent can add documents to memory, retrieve documents based on queries, and use the dependency graph for GASA.

### Model Integration

Models are integrated through the `ModelService`, which manages the LLM instance. The agent can use different model providers (OpenAI, Anthropic, vLLM, etc.) with a consistent interface.

## Extension Points

The Agent class is designed to be extensible:

### Custom Tools

You can create custom tools by implementing the `Tool` interface:

```python
from saplings.tools import Tool

class CustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="CustomTool",
            description="A custom tool for demonstration",
            parameters={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input for the tool",
                    }
                },
                "required": ["input"],
            }
        )

    def execute(self, input: str) -> Dict[str, Any]:
        """Execute the tool."""
        return {
            "result": f"Processed: {input}",
            "status": "success",
        }

# Register the tool with the agent
agent.register_tool(CustomTool())
```

### Custom Memory Store

You can create a custom memory store by implementing the `MemoryStore` interface:

```python
from saplings.memory import MemoryStore, Document, DocumentMetadata

class CustomMemoryStore(MemoryStore):
    def __init__(self, config=None):
        super().__init__(config)
        self.custom_index = {}

    async def add_document(self, content, metadata=None, document_id=None, embedding=None):
        """Add a document with custom indexing."""
        document = await super().add_document(content, metadata, document_id, embedding)

        # Add to custom index
        keywords = self._extract_keywords(content)
        for keyword in keywords:
            if keyword not in self.custom_index:
                self.custom_index[keyword] = []
            self.custom_index[keyword].append(document.id)

        return document

    def _extract_keywords(self, content):
        """Extract keywords from content."""
        # Simple keyword extraction
        words = content.lower().split()
        return [word for word in words if len(word) > 5]

    async def search_by_keyword(self, keyword):
        """Search documents by keyword."""
        keyword = keyword.lower()
        if keyword in self.custom_index:
            document_ids = self.custom_index[keyword]
            return [self.get_document(doc_id) for doc_id in document_ids]
        return []

# Use the custom memory store
custom_memory = CustomMemoryStore()
agent.memory_store = custom_memory
```

### Custom Agent Configuration

You can create a custom agent configuration by extending the `AgentConfig` class:

```python
from saplings.agent_config import AgentConfig

class CustomAgentConfig(AgentConfig):
    def __init__(
        self,
        provider: str,
        model_name: str,
        custom_setting: str = "default",
        **kwargs
    ):
        super().__init__(provider, model_name, **kwargs)
        self.custom_setting = custom_setting

        # Override default settings
        self.enable_gasa = True
        self.gasa_max_hops = 3
        self.retrieval_max_documents = 15

# Use the custom configuration
config = CustomAgentConfig(
    provider="openai",
    model_name="gpt-4o",
    custom_setting="specialized",
    temperature=0.5,
)
agent = Agent(config=config)
```

## Conclusion

The Agent class is the primary entry point for using Saplings, providing a clean, intuitive API that integrates all components of the framework. By using composition and dependency inversion, it offers a flexible and maintainable way to build powerful AI agents with rich capabilities like structural memory, cascaded retrieval, planning and execution, validation and self-improvement, tool integration, and multimodal support.
