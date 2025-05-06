# Saplings 🌱

A graph-first, self-improving agent framework that takes root in your repository or knowledge base, builds a structural map, and grows smarter each day through automated critique → fine-tune loops—all without locking you to a single model provider.

<img src="logo.png" alt="Saplings - A graph-first, self-improving agent framework for building AI agents" width="200">
- Provided by [Fine-Tuna](https://fine-tuna.ai): A dataset creation and fine-tuning agency.

## What is Saplings?

Saplings is a Python framework for building intelligent agents that can understand, reason about, and interact with complex information structures like codebases, knowledge bases, and document collections. Unlike traditional RAG (Retrieval-Augmented Generation) systems that treat documents as isolated chunks, Saplings builds and leverages structural relationships between information pieces, enabling more coherent and accurate reasoning.

The framework's name reflects its philosophy: every "sapling" starts small but takes root in your data, builds a structural map, and grows smarter through continuous learning and self-improvement.

Saplings is a lightweight (≤ 1.2k LOC core) framework for building domain-aware, self-critiquing autonomous agents with key pillars including structural memory, cascaded retrieval, guard-railed generation, self-improvement loops, and Graph-Aligned Sparse Attention (GASA) for faster, better-grounded reasoning.

## When to Use Saplings

Saplings is ideal for scenarios where:

- **Complex Information Structures**: You're working with interconnected information (codebases, technical documentation, research papers)
- **Contextual Understanding**: You need agents that understand relationships between different pieces of information
- **Long-Term Value**: You want agents that improve over time through usage
- **Customization**: You need fine-grained control over retrieval, reasoning, and generation
- **Cost Efficiency**: You want to optimize token usage and computational resources
- **Privacy**: You need control over how your data is processed and stored

### Ideal Use Cases

- **Code Understanding & Generation**: Analyzing codebases, generating documentation, suggesting improvements
- **Research Assistance**: Analyzing research papers, identifying connections, generating insights
- **Knowledge Management**: Organizing and retrieving information from complex knowledge bases
- **Technical Support**: Providing contextually aware answers to technical questions
- **Data Analysis**: Extracting insights from structured and unstructured data sources
- **Multi-step Reasoning**: Tasks requiring planning and sequential decision-making

## When NOT to Use Saplings

Saplings may not be the best choice when:

- **Simple Q&A**: For straightforward question-answering without context, simpler frameworks may suffice
- **Static Outputs**: When you need fixed, predictable outputs without dynamic decision-making
- **Minimal Context**: When context and relationships between information aren't important
- **Single-use Applications**: When you don't need agents that improve over time
- **Extremely Resource-constrained Environments**: The full framework may be overkill for very limited computing environments

For simple tasks with well-defined outputs, consider using:

- Direct API calls to language models
- Template-based generation systems
- Rule-based systems
- Traditional information retrieval

## Key Features

- **Structural Memory** — Vector + graph store per corpus with relationship tracking
- **Cascaded, Entropy-Aware Retrieval** — TF-IDF → embeddings → graph expansion for optimal context selection
- **Guard-railed Generation** — Planner with budget awareness, Executor with speculative draft/verify cycles
- **JudgeAgent & Validator Loop** — Reflexive scoring, self-healing patches for continuous improvement
- **Self-Healing & Adaptation** — Error analysis, automatic patching, and LoRA fine-tuning
- **Extensibility** — Hot-pluggable models, tools, validators with a unified interface
- **Dependency Injection** — Centralized service lifecycle and dependency management
- **Graph-Aligned Sparse Attention (GASA)** — Graph-conditioned attention masks for faster, better-grounded reasoning
- **Comprehensive Monitoring** — Tracing, visualization, and performance analysis tools
- **Multimodal Support** — Handling text, image, audio, and video content with a unified interface
- **Model Flexibility** — Support for OpenAI, Anthropic, vLLM, and HuggingFace models
- **Tool Integration** — Built-in tools and dynamic tool creation capabilities

## How It Differs From Existing Libraries

| Feature          | Saplings                                                        | LlamaIndex                                | LangChain                 | AutoGPT                |
| ---------------- | --------------------------------------------------------------- | ----------------------------------------- | ------------------------- | ---------------------- |
| Memory model     | Vector + dependency graph, MemoryStore                          | Vector stores (graph optional)            | Vector stores             | Short-term memory only |
| Context packing  | Graph-Aligned Sparse Attention (GASA) → 40% fewer FLOPs         | Dense concat or basic chunk window        | Dense concat              | Dense concat           |
| Self-improvement | Built-in JudgeAgent + Validator → LoRA adapters, self-healing   | Manual re-index / retrain only            | None                      | Limited                |
| Extensibility    | Plug-in adapters, indexers, validators, dynamic tool synthesis  | Integrations but no dynamic tool creation | Tool integrations         | Fixed tool set         |
| Cost governance  | Budget-aware planner, usage metering                            | None                                      | Basic tracking            | None                   |
| Privacy          | Configurable memory stores, local model support                 | Varies by vector DB                       | Depends on implementation | Limited                |
| Model support    | OpenAI, Anthropic, vLLM, HuggingFace, custom adapters           | Multiple providers                        | Multiple providers        | Primarily OpenAI       |
| Monitoring       | Comprehensive tracing, visualization, blame graph               | Basic                                     | Basic                     | Limited                |
| Multimodal       | Goal-specific agents with unified modality support              | Modality-specific agents                  | Modality-specific agents  | Limited                |
| GASA             | Full support with vLLM, optimized context with third-party APIs | Not available                             | Not available             | Not available          |

## Core Concepts

### Agent Architecture

Saplings agents are composed of several key components that work together to provide a comprehensive framework for building intelligent agents:

1. **Memory**: Stores and indexes documents, code, and other information
   - The `MemoryStore` combines vector storage and graph-based memory
   - Memory can be persisted to disk using `memory_store.save("path/to/directory")` and loaded with `memory_store.load("path/to/directory")`
   - When saved, it creates a directory structure with vector embeddings, dependency graph data, and configuration
2. **Retrieval**: Finds relevant information based on queries using a cascaded approach
   - **CascadeRetriever**: Orchestrates the entire retrieval pipeline
   - **TFIDFRetriever**: Performs initial filtering using TF-IDF
   - **EmbeddingRetriever**: Finds similar documents using embeddings
   - **GraphExpander**: Expands results using the dependency graph
3. **Planning**: Breaks down complex tasks into manageable steps
   - **SequentialPlanner**: Creates and optimizes execution plans
   - **PlanStep**: Represents a single step in a plan
   - **BudgetStrategy**: Manages resource allocation and constraints
4. **Execution**: Carries out individual steps using models and tools
   - **Executor**: Executes prompts with retrieved context
   - **RefinementStrategy**: Improves outputs through iterative refinement
   - **VerificationStrategy**: Verifies outputs against expectations
5. **Validation**: Ensures outputs meet quality standards
   - **ValidatorService**: Validates outputs against requirements
   - **JudgeAgent**: Evaluates output quality and provides feedback
6. **Tools**: Provides functionality for agents to interact with the world
   - **Tool**: Base class for all tools
   - **ToolRegistry**: Manages tool registration and discovery
   - **Default Tools**: Built-in tools for common tasks
   - **MCPClient**: Client for Machine Control Protocol servers
7. **Tool Factory**: Enables dynamic creation of tools
   - **ToolFactory**: Creates tools from specifications
   - **ToolValidator**: Validates generated tool code
8. **Monitoring**: Tracks agent performance and behavior
   - **TraceManager**: Manages execution traces
   - **Visualization**: Provides visual representations of performance

### Graph-Aligned Sparse Attention (GASA)

GASA is a novel technique that injects learned binary attention masks—derived from retrieval dependency graphs—into transformer layers. This allows:

- Full attention between tokens whose source chunks are within a defined number of hops in the graph
- Routing other tokens through lightweight global summary tokens
- Reducing computational costs while improving grounding
- Empirically fewer hallucinations in outputs

#### GASA with Third-Party LLMs

When using GASA with third-party APIs like OpenAI and Anthropic:

- Full GASA implementation is only possible with local models (vLLM) where we have direct access to the attention mechanism
- With third-party APIs, Saplings provides several alternative approaches:
  - **Shadow Model Tokenization**: Uses a small local model (default: Qwen/Qwen3-0.6B) for tokenization and mask generation
  - **Graph-Aware Prompt Composition**: Structures prompts based on graph relationships
  - **Block-Diagonal Packing**: Reorders chunks to create a block-diagonal structure
- The `enable_gasa` flag works with all model providers, automatically selecting the appropriate strategy

### Self-Improvement Loop

Saplings agents improve over time through:

1. **Performance Monitoring**: Tracking success rates, errors, and efficiency
2. **Validation**: Checking outputs against expected criteria
3. **Judgment**: Evaluating overall performance using JudgeAgent and identifying improvement areas
4. **Adaptation**: Automatically adjusting prompts, retrieval strategies, and other parameters
5. **Fine-tuning**: Creating specialized LoRA adapters for recurring tasks

### Multimodal Support

Saplings provides a flexible system for handling multiple modalities:

1. **Unified Interface**: Goal-specific agents that handle multiple modalities rather than separate agents for each modality
2. **Modality Handlers**: Specialized handlers for text, image, audio, and video content
3. **Flexible Input/Output**: Specify input and output modalities when running tasks
4. **Message Integration**: Seamless conversion between modalities and message content
5. **Extensibility**: Easy to add support for new modalities

## Installation

```bash
pip install saplings
```

For development installations with all optional dependencies:

```bash
pip install "saplings[dev,viz,monitoring,tools]"
```

### Running Tests

To run the standard test suite:

```bash
pytest
```

#### Benchmark Tests

Benchmark tests are excluded from the standard test suite because they can be resource-intensive and may occasionally hang. To run benchmark tests:

```bash
# Run all tests including benchmarks
pytest --run-benchmarks

# Run only benchmark tests
pytest tests/benchmarks/ --run-benchmarks

# Run a specific benchmark test
pytest tests/benchmarks/test_gasa_benchmark.py::TestGASABenchmark::test_gasa_flop_reduction --run-benchmarks
```

Note: Benchmark tests include timeouts to prevent hanging, but they may still be unstable in some environments.

## API Key Setup

Saplings supports multiple model providers, each requiring their own API keys. Here's how to set them up:

### Setting Environment Variables

The recommended way to provide API keys is through environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_api_key

# For Anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key

# For HuggingFace Hub (if using their API)
export HUGGINGFACE_API_KEY=your_huggingface_api_key
```

### Using a .env File

You can also use a `.env` file in your project root:

```
# .env file
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

Then load it in your Python code:

```python
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
```

### Specifying API Keys in Code

While not recommended for production, you can also specify API keys directly when creating the model:

```python
# OpenAI
model = LLM.create(provider="openai", model_name="gpt-4o", api_key="your_openai_api_key")

# Anthropic
model = LLM.create(provider="anthropic", model_name="claude-3-opus-20240229", api_key="your_anthropic_api_key")
```

## Default Tools

Saplings provides a set of ready-to-use tools that can be easily integrated into your agents:

### Available Tools

| Tool                    | Description                                             | Dependencies                       |
| ----------------------- | ------------------------------------------------------- | ---------------------------------- |
| `PythonInterpreterTool` | Executes Python code in a sandboxed environment         | None                               |
| `FinalAnswerTool`       | Provides a final answer to a problem                    | None                               |
| `UserInputTool`         | Gets input from the user                                | None                               |
| `DuckDuckGoSearchTool`  | Performs web searches using DuckDuckGo                  | `duckduckgo-search`                |
| `GoogleSearchTool`      | Performs web searches using Google                      | `requests`                         |
| `VisitWebpageTool`      | Visits a webpage and returns its content as markdown    | `markdownify`, `requests`          |
| `WikipediaSearchTool`   | Searches Wikipedia and returns article content          | `wikipedia-api`                    |
| `SpeechToTextTool`      | Transcribes audio to text                               | `transformers`, `torch`, `librosa` |
| `MCPClient`             | Connects to MCP servers and makes their tools available | `mcpadapt`                         |

### Using Default Tools

```python
from saplings import Agent, AgentConfig
from saplings.tools import PythonInterpreterTool, DuckDuckGoSearchTool, WikipediaSearchTool

# Create tools
python_tool = PythonInterpreterTool()
search_tool = DuckDuckGoSearchTool(max_results=5)
wiki_tool = WikipediaSearchTool(user_agent="YourApp/1.0")

# Create an agent with the tools
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        tools=[python_tool, search_tool, wiki_tool]
    )
)

# Run a task that uses the tools
import asyncio
result = asyncio.run(agent.run(
    "Find information about neural networks on Wikipedia, then write a Python function to create a simple neural network structure."
))
```

### Helper Functions

Saplings provides helper functions to easily access default tools:

```python
from saplings.tools import get_default_tool, get_all_default_tools

# Get a specific tool
python_tool = get_default_tool("python_interpreter")

# Get all default tools
all_tools = get_all_default_tools()
```

### Using MCP Client

The MCP (Machine Control Protocol) client allows Saplings agents to use tools from MCP servers:

```python
from saplings import Agent, AgentConfig
from saplings.tools import MCPClient

# Connect to an MCP server
with MCPClient({"command": "path/to/mcp/server"}) as mcp_tools:
    # Create an agent with the MCP tools
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=mcp_tools
        )
    )

    # Run a task that uses the MCP tools
    import asyncio
    result = asyncio.run(agent.run(
        "Use the MCP tools to perform a task."
    ))
```

You can also connect to multiple MCP servers at once:

```python
server_parameters = [
    {"command": "path/to/first/mcp/server"},
    {"url": "http://localhost:8000/sse"},  # SSE server
]

with MCPClient(server_parameters) as mcp_tools:
    # Create an agent with tools from both MCP servers
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            tools=mcp_tools
        )
    )
```

### Installing Dependencies

To use all default tools, install the required dependencies:

```bash
pip install "saplings[tools]"
```

For MCP support:

```bash
pip install "saplings[mcp]"
```

Or install specific dependencies as needed:

```bash
pip install duckduckgo-search markdownify wikipedia-api mcpadapt
```

## Model Providers

Saplings supports multiple model providers that can be easily switched:

### Available Providers

| Provider    | Description                              | Example                              |
| ----------- | ---------------------------------------- | ------------------------------------ |
| OpenAI      | OpenAI API models (GPT-4, GPT-3.5, etc.) | `"gpt-4o"`, `"gpt-4-turbo"`          |
| Anthropic   | Anthropic API models (Claude 3, etc.)    | `"claude-3-opus-20240229"`           |
| vLLM        | Local models served through vLLM         | `"meta-llama/Llama-3.1-8B-Instruct"` |
| HuggingFace | Models from Hugging Face                 | `"meta-llama/Llama-3-8b-instruct"`   |

### Changing Providers

You can easily switch between providers by specifying the provider and model name:

```python
# Using OpenAI
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o"
)

# Using Anthropic
config = AgentConfig(
    provider="anthropic",
    model_name="claude-3-opus-20240229"
)

# Using vLLM (local)
config = AgentConfig(
    provider="vllm",
    model_name="meta-llama/Llama-3.1-8B-Instruct"
)
```

### Provider-Specific Parameters

Each provider supports additional parameters:

```python
# OpenAI with parameters
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=1024
)

# Anthropic with parameters
config = AgentConfig(
    provider="anthropic",
    model_name="claude-3-opus-20240229",
    temperature=0.5,
    max_tokens=2048
)

# vLLM with parameters
config = AgentConfig(
    provider="vllm",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.8,
    quantization="awq"
)
```

### Model Registry

Saplings includes a model registry that ensures only one instance of a model with the same configuration is created. This helps reduce memory usage and improve performance, especially when using multiple agents with the same model.

The model registry is enabled by default and works automatically. When you create multiple agents with the same model configuration, they will share a single model instance:

```python
# Create two agents with the same model configuration
agent1 = Agent(AgentConfig(provider="openai", model_name="gpt-4o"))
agent2 = Agent(AgentConfig(provider="openai", model_name="gpt-4o"))

# Both agents will use the same model instance
# This saves memory and improves performance
```

You can disable the model registry by setting the `SAPLINGS_ENABLE_MODEL_REGISTRY` environment variable:

```bash
# Disable the model registry
export SAPLINGS_ENABLE_MODEL_REGISTRY=0
```

This is particularly useful for:

- Multi-agent systems where several agents use the same model
- Applications with multiple components that need access to the same model
- Reducing VRAM usage when running local models like vLLM

## Quick Start

### Basic Agent

```python
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
            memory_path="./agent_memory"  # Directory where memory will be saved/loaded
        )
    )

    # Set the memory store
    agent.memory_store = memory

    # Memory will be automatically saved to the memory_path directory
    # You can also manually save/load memory:
    # memory.save("./custom_memory_path")
    # memory.load("./custom_memory_path")

    # Run a simple query
    result = await agent.run("What is Saplings?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Agent with GASA

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph

async def main():
    # Initialize memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Add documents to memory
    await memory.add_document(
        "Graph-Aligned Sparse Attention (GASA) is a technique that improves efficiency and grounding in language models by focusing attention on relevant context based on document relationships.",
        metadata={"source": "docs.txt", "section": "gasa"}
    )

    await memory.add_document(
        "GASA injects a binary attention mask—derived from the retrieval dependency graph—into each transformer layer, permitting full attention only between tokens whose source chunks are within a defined number of hops in the graph.",
        metadata={"source": "docs.txt", "section": "gasa_details"}
    )

    # Build dependency graph
    await graph.build_from_memory(memory)

    # Create agent configuration with GASA enabled
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,
            gasa_max_hops=2,
            gasa_strategy="binary",
            gasa_fallback="prompt_composer",
            gasa_shadow_model=True,
            gasa_shadow_model_name="Qwen/Qwen3-0.6B",
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Run a query
    result = await agent.run("Explain how GASA works")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Agent with Tools

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
```

### Self-Hosted Model with vLLM

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore

async def main():
    # Create a memory store
    memory = MemoryStore()

    # Add some documents
    await memory.add_document(
        "Saplings is a graph-first, self-improving agent framework that takes root in your repository.",
        metadata={"type": "documentation", "section": "overview"}
    )

    # Create an agent with self-hosted model
    agent = Agent(
        config=AgentConfig(
            provider="vllm",
            model_name="Qwen/Qwen3-1.7B",
            # vLLM specific parameters
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            quantization="awq",
            # Enable GASA - fully supported with vLLM
            enable_gasa=True,
            gasa_max_hops=2,
        )
    )

    # Set the memory store
    agent.memory_store = memory

    # Run a query
    result = await agent.run("Explain what Saplings is")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Multimodal Agent

```python
import asyncio
from saplings import Agent, AgentConfig

async def main():
    # Create agent configuration with multimodal support
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            memory_path="./agent_memory",
            supported_modalities=["text", "image"],  # Specify supported modalities
        )
    )

    # Run a task with specific input and output modalities
    result = await agent.run(
        task="Generate a description of a sunset over mountains and create an image of it.",
        input_modalities=["text"],
        output_modalities=["text", "image"]
    )

    # Access the text output
    text_output = result.get("text")
    print(text_output)

    # Access the image output (if available)
    image_output = result.get("image")
    if image_output:
        print(f"Image generated: {image_output}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Multi-Agent Orchestration

Saplings supports orchestrating multiple specialized agents that work together:

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore
from saplings.orchestration import GraphRunner, AgentNode
from saplings.core.model_adapter import LLM

async def main():
    # Create a model for orchestration
    model = LLM.create(provider="openai", model_name="gpt-4o")

    # Create a graph runner for agent coordination
    graph_runner = GraphRunner(model=model)

    # Create specialized agents
    code_analyzer = Agent(AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        name="CodeAnalyzer",
    ))
    code_analyzer.memory_store = MemoryStore()

    refactoring_expert = Agent(AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        name="RefactoringExpert",
    ))
    refactoring_expert.memory_store = MemoryStore()

    documentation_writer = Agent(AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        name="DocumentationWriter",
    ))
    documentation_writer.memory_store = MemoryStore()

    # Register agents with the graph runner
    graph_runner.register_agent(AgentNode(
        id="code_analyzer",
        name="Code Analyzer",
        description="Analyzes code structure and identifies patterns",
        agent=code_analyzer
    ))

    graph_runner.register_agent(AgentNode(
        id="refactoring_expert",
        name="Refactoring Expert",
        description="Suggests and implements code refactoring",
        agent=refactoring_expert
    ))

    graph_runner.register_agent(AgentNode(
        id="documentation_writer",
        name="Documentation Writer",
        description="Creates clear, comprehensive documentation",
        agent=documentation_writer
    ))

    # Run a debate between agents
    result = await graph_runner.run_debate(
        task="Improve the error handling in auth.py and update the documentation",
        agent_ids=["code_analyzer", "refactoring_expert", "documentation_writer"]
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Custom Model Adapters

```python
import asyncio
from saplings.core.model_adapter import LLM
from saplings import Agent, AgentConfig

async def main():
    # Use a specific model
    openai_model = LLM.create(provider="openai", model_name="gpt-4o")
    agent = Agent(AgentConfig(provider="openai", model_name="gpt-4o"))

    # Use a local model with vLLM
    vllm_model = LLM.create(
        provider="vllm",
        model_name="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8
    )

    # Create a custom adapter by inheriting from LLM
    class MyCustomAdapter(LLM):
        def __init__(self, provider: str, model_name: str, **kwargs):
            self.provider = provider
            self.model_name = model_name
            # Custom initialization

        async def generate(self, prompt, **kwargs):
            # Custom implementation
            pass

        async def generate_streaming(self, prompt, **kwargs):
            # Custom implementation
            pass

        @property
        def metadata(self):
            # Return metadata about the model
            return {
                "provider": self.provider,
                "model_name": self.model_name,
                "context_window": 4096,
                "capabilities": ["text-generation"]
            }

    # Use the custom adapter
    custom_model = MyCustomAdapter(provider="custom", model_name="my-model")
    agent = Agent(AgentConfig(provider="custom", model_name="my-model"))
    agent.model = custom_model

    # Run a task with the custom model
    result = await agent.run("Explain what a custom model adapter is")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Dynamic Tool Creation

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.core.model_adapter import LLM

async def main():
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
    visualization_tool = await agent.create_tool(
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
    )

    # Register the tool with the agent
    agent.register_tool(visualization_tool)

    # Use the agent with the new tool
    result = await agent.run(
        "Create a visualization of the following data points: x=[1,2,3,4,5], y=[10,25,15,30,20]"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Self-Improvement

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.judge import JudgeAgent
from saplings.validator import ValidatorRegistry
from saplings.core.model_adapter import LLM

async def main():
    # Create a model
    model = LLM.create(provider="openai", model_name="gpt-4o")

    # Create an agent with self-healing enabled
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_self_healing=True,
            self_healing_max_retries=3,
        )
    )

    # Create judge and validator components
    judge = JudgeAgent(model=model)
    validator_registry = ValidatorRegistry()

    # Run a task
    result = await agent.run("Implement a sorting algorithm")

    # Get the output from the result
    output = result.get("output", "")

    # Validate the result
    validator = validator_registry.get_validator("code_validator")
    validation = await validator.validate(
        output=output,
        prompt="Implement a sorting algorithm"
    )

    # Judge the performance
    judgment = await judge.judge(
        output=output,
        prompt="Implement a sorting algorithm"
    )

    # Improve the agent
    improvement_plan = await agent.self_improve(
        validation_results=[validation],
        judgment_results=[judgment]
    )

    print(f"Improvement plan: {improvement_plan}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Project Roadmap

The Saplings framework is under active development with the following roadmap:

### Short-term

- **GASA Enhancements**
  - Improved shadow model tokenization for third-party LLMs
  - Multi-head attention with different masks for different heads
  - Dynamic masks that change during generation based on context
- **Retrieval Improvements**
  - Learning-to-rank techniques for improved document ranking
  - Query expansion for better recall
  - Personalized retrieval based on user preferences
- **Model Adapters**
  - Enhanced vLLM integration with all advanced features
  - Support for more vision-language models
  - Improved function calling across all model providers

### Mid-term

- **Memory and Storage**
  - Distributed vector stores and graph databases
  - Incremental indexing for efficient document updates
  - Advanced entity linking and resolution
  - Temporal graphs with time-aware relationships
- **Self-Improvement**
  - Automated LoRA fine-tuning pipelines
  - Improved patch generation with static analysis
  - Sandboxed execution for patch validation
- **Executor and Planning**
  - ValidatorRegistry for specialized verification tasks
  - Multi-model verification for improved accuracy
  - Adaptive refinement strategies based on verification results

### Long-term

- **Multi-modal Support**
  - Memory stores for images, audio, and other modalities
  - Multi-modal retrieval and reasoning
  - Cross-modal relationship tracking
- **Advanced Orchestration**
  - Hierarchical agent structures with specialized roles
  - Improved negotiation strategies for multi-agent systems
  - Automated agent specialization based on performance
- **Privacy and Security**
  - Differential privacy guarantees for sensitive data
  - Federated learning for privacy-preserving model improvement
  - Enhanced security auditing and verification

## Documentation

For detailed documentation, see the [docs](./docs) directory:

- [Quick Start Guide](./docs/quick_start.md)
- [Core Concepts](./docs/core_concepts.md)
- [Memory and Retrieval](./docs/memory.md)
- [GASA Implementation](./docs/gasa.md)
- [GASA with Third-Party LLMs](./docs/gasa_third_party.md)
- [Model Adapters](./docs/model_adapters.md)
- [Agent](./docs/agent.md)
- [Executor and Planning](./docs/executor.md)
- [Tools](./docs/tools.md)
- [Validation and Judging](./docs/validation.md)
- [Multimodal Support](./docs/multimodal.md)
- [Examples](./docs/examples.md)

## Development

### Prerequisites

- Python 3.9+
- Poetry (optional, for development)

### Setup

```bash
# Install from PyPI
pip install saplings

# For development installation
git clone https://github.com/jacobwarren/saplings.git
cd saplings
pip install -e ".[dev]"

# Run tests
pytest
```

### Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Community and Support

- [GitHub Issues](https://github.com/jacobwarren/saplings/issues): Bug reports and feature requests
- [Discussions](https://github.com/jacobwarren/saplings/discussions): Questions and community discussions

## License

MIT

## Citation

If you use Saplings in your research, please cite:

```bibtex
@software{saplings2023,
  author = {Jacob Warren},
  title = {Saplings: A Graphs-first, Self-improving Agent Framework},
  url = {https://github.com/jacobwarren/saplings},
  year = {2023},
}
```

Or use the badge:

![Static Badge](https://img.shields.io/badge/built_with_Saplings-a?style=flat&label=%F0%9F%8C%B1&labelColor=%23666666&color=%231D8039&cacheSeconds=3600)
