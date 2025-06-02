# Saplings - AI Agent Framework

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Saplings is a powerful, graphs-first, self-improving AI agent framework for building intelligent applications. It provides a comprehensive suite of tools for creating AI agents with advanced capabilities including memory management, tool integration, multi-modal processing, and sophisticated attention mechanisms.

## 🚀 Key Features

- **Universal LLM Support**: Works with OpenAI, Anthropic, vLLM, and HuggingFace models
- **Graph-Aligned Sparse Attention (GASA)**: Advanced attention mechanism for improved performance
- **Self-Healing Capabilities**: Automatic error recovery and performance optimization
- **Rich Tool Ecosystem**: Built-in tools for web search, code execution, file operations, and more
- **Multi-Modal Support**: Handle text, images, audio, and video inputs
- **Memory & Retrieval**: Persistent memory with intelligent document retrieval
- **Planning & Execution**: Multi-step task planning with budget management
- **Monitoring & Validation**: Comprehensive logging, tracing, and output validation
- **Dependency Injection**: Clean architecture with configurable services

## 📦 Installation

### Basic Installation

```bash
pip install saplings
```

### Installation with Extras

```bash
# Full installation with all features
pip install saplings[full]

# Specific feature sets
pip install saplings[transformers,tools,viz]  # Transformers + tools + visualization
pip install saplings[retrieval,faiss]         # Enhanced retrieval with FAISS
pip install saplings[browser,mcp]             # Browser automation + MCP tools
pip install saplings[lora]                    # LoRA fine-tuning support
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/jacobwarren/saplings
cd saplings

# Install with Poetry (recommended)
poetry install --extras full

# Or with pip
pip install -e .[dev,full]
```

## 🏃‍♂️ Quick Start

### Simple Agent Creation

```python
from saplings import Agent, AgentConfig

# Create an agent with minimal configuration
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-openai-api-key"
)

# Run a simple task
result = await agent.run("What is the capital of France?")
print(result)  # "The capital of France is Paris."
```

### Synchronous Usage

```python
# For synchronous environments
agent = Agent(provider="openai", model_name="gpt-4o")
result = agent.run_sync("Calculate the factorial of 5")
print(result)
```

### Using Configuration Presets

```python
# Quick setup for specific providers
agent = Agent.from_config(
    AgentConfig.for_openai("gpt-4o", api_key="your-key")
)

# Or for Anthropic
agent = Agent.from_config(
    AgentConfig.for_anthropic("claude-3-opus", api_key="your-key")
)

# Or for local vLLM
agent = Agent.from_config(
    AgentConfig.for_vllm("Qwen/Qwen3-7B-Instruct")
)
```

## 🔧 Advanced Configuration

### Builder Pattern

```python
from saplings import AgentBuilder

agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_memory_path("./agent_memory") \
    .with_output_dir("./agent_output") \
    .with_gasa_enabled(True) \
    .with_monitoring_enabled(True) \
    .with_self_healing_enabled(True) \
    .with_tools([
        "PythonInterpreterTool",
        "DuckDuckGoSearchTool",
        "WikipediaSearchTool"
    ]) \
    .with_model_parameters({
        "temperature": 0.7,
        "max_tokens": 2048,
    }) \
    .build()
```

### Custom Configuration

```python
from saplings import AgentConfig

config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-key",
    
    # Advanced features
    enable_gasa=True,
    gasa_max_hops=3,
    gasa_strategy="binary",
    gasa_fallback="prompt_composer",
    
    # Self-healing
    enable_self_healing=True,
    self_healing_max_retries=3,
    
    # Memory & retrieval
    memory_path="./agent_memory",
    retrieval_max_documents=20,
    retrieval_entropy_threshold=0.1,
    
    # Planning
    planner_budget_strategy="dynamic",
    planner_total_budget=2.0,
    planner_allow_budget_overflow=True,
    
    # Tool factory
    enable_tool_factory=True,
    tool_factory_sandbox_enabled=True,
    allowed_imports=["os", "json", "math", "numpy", "pandas"],
    
    # Multi-modal support
    supported_modalities=["text", "image", "audio"],
    
    # Monitoring
    enable_monitoring=True
)

agent = Agent(config=config)
```

## 🛠️ Tools and Extensions

### Built-in Tools

```python
from saplings import Agent
from saplings.api.tools import (
    PythonInterpreterTool,
    DuckDuckGoSearchTool,
    WikipediaSearchTool,
    GoogleSearchTool,
    VisitWebpageTool,
    UserInputTool,
    FinalAnswerTool
)

# Create agent with specific tools
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    tools=[
        PythonInterpreterTool(),
        DuckDuckGoSearchTool(),
        WikipediaSearchTool(),
        VisitWebpageTool()
    ]
)

# Use the agent for complex tasks
result = await agent.run(
    "Search for information about quantum computing and create a Python script to demonstrate quantum superposition"
)
```

### Custom Tools

```python
from saplings.api.tools import tool

@tool(name="calculator", description="Performs mathematical calculations")
def calculate(expression: str) -> float:
    """
    Calculate the result of a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The numerical result of the calculation
    """
    # Safe evaluation of mathematical expressions
    import ast
    import operator
    
    # Define allowed operations
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    
    def eval_expr(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    return eval_expr(ast.parse(expression, mode='eval').body)

# Register the tool
agent.register_tool(calculate)
```

### Browser Tools

```python
from saplings.api.browser_tools import BrowserManager
from saplings import Agent

# Enable browser tools (requires selenium)
browser_manager = BrowserManager(headless=False)
browser_manager.initialize()

agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    tools=[
        "GoToTool",
        "ClickTool", 
        "TypeTool",
        "ScrollTool",
        "ScreenshotTool"
    ]
)

# Use for web automation tasks
result = await agent.run(
    "Go to example.com, take a screenshot, and summarize the main content"
)

browser_manager.close()
```

## 💾 Memory and Document Management

### Adding Documents

```python
# Add individual documents
agent.add_document(
    content="Python is a high-level programming language...",
    metadata={"source": "tutorial", "topic": "programming"}
)

# Add documents from files
agent.add_documents_from_directory(
    directory="./docs",
    extension=".md"
)

# Add documents from URLs
await agent.add_document_from_url("https://example.com/article")
```

### Retrieval and Context

```python
# Retrieve relevant documents
relevant_docs = await agent.retrieve(
    query="machine learning algorithms",
    limit=5,
    fast_mode=True
)

# Use retrieved context in tasks
result = await agent.run(
    task="Explain the differences between supervised and unsupervised learning",
    context=relevant_docs,
    skip_retrieval=False  # Still allow additional retrieval
)
```

## 🎯 Advanced Features

### Graph-Aligned Sparse Attention (GASA)

GASA is a novel attention mechanism that improves performance by focusing on relevant parts of the input:

```python
config = AgentConfig.for_openai(
    "gpt-4o",
    # GASA configuration
    enable_gasa=True,
    gasa_max_hops=3,
    gasa_strategy="binary",           # binary, soft, learned
    gasa_fallback="prompt_composer",  # block_diagonal, prompt_composer
    gasa_shadow_model=True,
    gasa_shadow_model_name="Qwen/Qwen3-0.6B",
    gasa_prompt_composer=True
)
```

### Self-Healing

Automatic error recovery and performance optimization:

```python
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    enable_self_healing=True,
    self_healing_max_retries=3
)

# The agent will automatically retry failed operations
# and learn from mistakes to improve future performance
result = await agent.run("Complex task that might fail initially")
```

### Monitoring and Tracing

```python
from saplings import Agent

agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    enable_monitoring=True
)

# Run tasks with full tracing
result = await agent.run("Analyze this data", save_results=True)

# Access monitoring data
traces = agent.get_execution_traces()
metrics = agent.get_performance_metrics()
```

### Multi-Modal Processing

```python
# Configure for multi-modal support
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    supported_modalities=["text", "image", "audio"]
)

# Process different input types
result = await agent.run(
    task="Describe this image and transcribe any text in it",
    input_modalities=["image"],
    context=[{"type": "image", "path": "./image.jpg"}]
)
```

### Planning and Execution

```python
# Enable advanced planning
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    planner_budget_strategy="dynamic",
    planner_total_budget=5.0,  # $5 USD budget
    planner_allow_budget_overflow=True,
    planner_budget_overflow_margin=0.1
)

agent = Agent(config=config)

# Create explicit plans
plan = await agent.plan(
    task="Research renewable energy trends and create a comprehensive report",
    context=[]
)

# Execute with plan
result = await agent.execute_plan(
    plan=plan,
    context=[],
    use_tools=True
)
```

## 🔍 Validation and Quality Control

```python
from saplings.api.validator import CodeValidator, FactualValidator

# Configure validation
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    executor_validation_type="judge",  # basic, execution, judge
    validators=[
        CodeValidator(),
        FactualValidator()
    ]
)

# Validate outputs
result = await agent.run("Write a Python function to sort a list")
validation_result = await agent.judge_output(
    input_data="Write a Python function to sort a list",
    output_data=result,
    judgment_type="code_quality"
)
```

## 🏗️ Architecture and Services

Saplings uses a clean architecture with dependency injection:

```python
from saplings import AgentFacade, AgentFacadeBuilder
from saplings.api.services import (
    MemoryService,
    RetrievalService,
    PlannerService,
    ExecutionService,
    ValidationService,
    ToolService,
    MonitoringService
)

# Custom service configuration
facade = AgentFacadeBuilder() \
    .with_config(config) \
    .with_memory_service(MemoryService()) \
    .with_retrieval_service(RetrievalService()) \
    .with_planner_service(PlannerService()) \
    .with_execution_service(ExecutionService()) \
    .with_validation_service(ValidationService()) \
    .with_tool_service(ToolService()) \
    .with_monitoring_service(MonitoringService()) \
    .build()
```

## 📊 Examples

### Example 1: Research Assistant

```python
from saplings import Agent, AgentConfig
from saplings.api.tools import DuckDuckGoSearchTool, WikipediaSearchTool, PythonInterpreterTool

# Create a research-focused agent
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    tools=[
        DuckDuckGoSearchTool(),
        WikipediaSearchTool(),
        PythonInterpreterTool()
    ],
    enable_gasa=True,
    enable_monitoring=True
)

# Perform research task
result = await agent.run("""
Research the latest developments in quantum computing in 2024. 
Provide a summary with key breakthroughs, major companies involved, 
and create a Python script to visualize the timeline of developments.
""")

print(result)
```

### Example 2: Code Analysis Agent

```python
from saplings import Agent
from saplings.api.tools import PythonInterpreterTool

agent = Agent(
    provider="anthropic",
    model_name="claude-3-opus",
    tools=[PythonInterpreterTool()],
    enable_tool_factory=True,
    tool_factory_sandbox_enabled=True
)

# Add codebase to memory
agent.add_documents_from_directory("./src", extension=".py")

# Analyze code
result = await agent.run("""
Analyze the codebase in memory and:
1. Identify potential performance bottlenecks
2. Suggest code quality improvements
3. Create unit tests for the main functions
4. Generate documentation for undocumented functions
""")
```

### Example 3: Multi-Modal Content Creator

```python
from saplings import Agent

agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    supported_modalities=["text", "image"],
    tools=["PythonInterpreterTool", "VisitWebpageTool"]
)

# Create content from multiple sources
result = await agent.run(
    task="""
    Create a comprehensive blog post about sustainable energy:
    1. Research current trends online
    2. Generate relevant charts and graphs
    3. Create accompanying images if needed
    4. Format as markdown with proper structure
    """,
    output_modalities=["text", "image"]
)
```

### Example 4: Data Analysis Pipeline

```python
from saplings import AgentBuilder

agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_tools([
        "PythonInterpreterTool",
        "DuckDuckGoSearchTool"
    ]) \
    .with_memory_path("./data_analysis_memory") \
    .with_monitoring_enabled(True) \
    .build()

# Load data into memory
agent.add_documents_from_directory("./datasets", extension=".csv")

# Perform analysis
result = await agent.run("""
Analyze the datasets in memory:
1. Perform exploratory data analysis
2. Identify patterns and correlations
3. Create visualizations
4. Generate insights and recommendations
5. Create a summary report
""")
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests

# Run with coverage
pytest --cov=saplings --cov-report=html
```

## 📚 API Reference

### Core Classes

- **`Agent`**: Main agent class for task execution
- **`AgentConfig`**: Configuration for agent behavior
- **`AgentBuilder`**: Builder pattern for agent creation
- **`AgentFacade`**: Service-oriented facade (beta)

### Tools

- **`PythonInterpreterTool`**: Execute Python code safely
- **`DuckDuckGoSearchTool`**: Web search functionality
- **`WikipediaSearchTool`**: Wikipedia search and retrieval
- **`VisitWebpageTool`**: Web page content extraction
- **`UserInputTool`**: Interactive user input
- **`FinalAnswerTool`**: Provide final responses

### Services

- **Memory Services**: Document storage and management
- **Retrieval Services**: Intelligent document retrieval
- **Planning Services**: Multi-step task planning
- **Execution Services**: Task execution and orchestration
- **Validation Services**: Output quality control
- **Monitoring Services**: Performance tracking and logging

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install for development
git clone https://github.com/jacobwarren/saplings
cd saplings
poetry install --extras dev

# Run pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [https://github.com/jacobwarren/saplings/docs](https://github.com/jacobwarren/saplings/docs)
- **Repository**: [https://github.com/jacobwarren/saplings](https://github.com/jacobwarren/saplings)
- **Issues**: [https://github.com/jacobwarren/saplings/issues](https://github.com/jacobwarren/saplings/issues)

## 🙏 Acknowledgments

Built with love for the AI community. Special thanks to all contributors and the open-source libraries that make this possible.