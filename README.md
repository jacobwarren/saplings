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
- **Fluent Builder API**: Discoverable, type-safe configuration through builder patterns

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

### Using the Builder API (Recommended)

The builder pattern is the primary and recommended way to create agents:

```python
import asyncio
from saplings import AgentBuilder

async def main():
    # Create an agent using the fluent builder API
    agent = AgentBuilder() \
        .with_provider("openai") \
        .with_model_name("gpt-4o") \
        .with_api_key("your-openai-api-key") \
        .build()
    
    # Run a simple task
    result = await agent.run("What is the capital of France?")
    print(result)  # "The capital of France is Paris."

asyncio.run(main())
```

### Using Configuration Presets

For common scenarios, use the preset factory methods:

```python
from saplings import AgentBuilder

# Quick setup for specific providers
agent = AgentBuilder.for_openai("gpt-4o") \
    .with_api_key("your-key") \
    .build()

# Or for Anthropic
agent = AgentBuilder.for_anthropic("claude-3-opus") \
    .with_api_key("your-key") \
    .build()

# Or for local vLLM
agent = AgentBuilder.for_vllm("Qwen/Qwen3-7B-Instruct") \
    .build()
```

### Convenience Agent Constructor

For simple use cases, you can also use the direct constructor:

```python
from saplings import Agent

# Direct instantiation (convenience method)
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-openai-api-key"
)

result = await agent.run("What is the capital of France?")
```

### Synchronous Usage

For non-async environments:

```python
from saplings import AgentBuilder

agent = AgentBuilder.for_openai("gpt-4o").build()
result = agent.run_sync("Calculate the factorial of 5")
print(result)
```

## 🔧 Configuration

Saplings offers flexible configuration through multiple approaches. For detailed configuration options, see the **[Configuration Guide](CONFIGURATION_GUIDE.md)**.

### Configuration Presets

Choose from predefined configurations optimized for different use cases:

| Preset | Use Case | Features | Resource Usage |
|--------|----------|----------|----------------|
| **Minimal** | Learning, simple tasks | Basic features only | Low |
| **Standard** | Most applications | Balanced feature set | Moderate |
| **Full-Featured** | Complex workflows | All advanced features | High |

```python
from saplings import AgentBuilder

# Choose the right preset for your needs
agent = AgentBuilder.minimal("openai", "gpt-4o").build()      # Simple tasks
agent = AgentBuilder.standard("openai", "gpt-4o").build()     # Most use cases  
agent = AgentBuilder.full_featured("openai", "gpt-4o").build() # Complex workflows
```

### Provider-Specific Optimizations

Each provider has optimized settings:

```python
# OpenAI optimizations (GASA shadow model, prompt composer)
agent = AgentBuilder.for_openai("gpt-4o").build()

# Anthropic optimizations (Constitutional AI alignment)  
agent = AgentBuilder.for_anthropic("claude-3-opus").build()

# vLLM optimizations (local model efficiency)
agent = AgentBuilder.for_vllm("Qwen/Qwen3-7B-Instruct").build()
```

### Comprehensive Builder Configuration

The builder API makes all configuration options discoverable:

```python
from saplings import AgentBuilder

agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_api_key("your-key") \
    .with_memory_path("./agent_memory") \
    .with_output_dir("./agent_output") \
    .with_gasa_enabled(True) \
    .with_gasa_max_hops(3) \
    .with_gasa_strategy("binary") \
    .with_gasa_fallback("prompt_composer") \
    .with_monitoring_enabled(True) \
    .with_self_healing_enabled(True) \
    .with_self_healing_max_retries(3) \
    .with_tool_factory_enabled(True) \
    .with_tools([
        "PythonInterpreterTool",
        "DuckDuckGoSearchTool",
        "WikipediaSearchTool"
    ]) \
    .with_planner_budget_strategy("dynamic") \
    .with_planner_total_budget(5.0) \
    .with_retrieval_max_documents(20) \
    .with_model_parameters({
        "temperature": 0.7,
        "max_tokens": 2048,
    }) \
    .with_supported_modalities(["text", "image", "audio"]) \
    .build()
```

### Configuration with Custom Services (Advanced)

For advanced use cases requiring custom service implementations:

```python
from saplings import AgentConfig, AgentFacadeBuilder
from saplings.api.services import CustomMemoryService, CustomToolService

# Create configuration
config = AgentConfig.for_openai("gpt-4o", api_key="your-key")

# Build agent with custom services
agent_facade = AgentFacadeBuilder() \
    .with_config(config) \
    .with_memory_service(CustomMemoryService()) \
    .with_tool_service(CustomToolService()) \
    .build()
```

> **📖 For comprehensive configuration documentation, see the [Configuration Guide](CONFIGURATION_GUIDE.md)**

## 🛠️ Tools and Extensions

### Built-in Tools with Builder

```python
import asyncio
from saplings import AgentBuilder

async def tool_example():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tools([
            "PythonInterpreterTool",
            "DuckDuckGoSearchTool", 
            "WikipediaSearchTool",
            "VisitWebpageTool"
        ]) \
        .build()
    
    # Complex task using multiple tools
    result = await agent.run("""
    Search for information about quantum computing and create a Python script 
    to demonstrate quantum superposition concepts
    """)
    
    print(result)

asyncio.run(tool_example())
```

### Custom Tools

```python
from saplings import AgentBuilder
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

# Build agent with custom tool
agent = AgentBuilder.for_openai("gpt-4o") \
    .with_custom_tools([calculate]) \
    .build()
```

### Browser Tools

```python
from saplings import AgentBuilder
from saplings.api.browser_tools import BrowserManager

async def browser_example():
    # Initialize browser manager
    browser_manager = BrowserManager(headless=False)
    browser_manager.initialize()
    
    try:
        agent = AgentBuilder.for_openai("gpt-4o") \
            .with_browser_tools_enabled(True) \
            .with_tools([
                "GoToTool",
                "ClickTool", 
                "TypeTool",
                "ScreenshotTool"
            ]) \
            .build()
        
        result = await agent.run("""
        Go to example.com, take a screenshot, 
        and summarize the main content
        """)
        
        print(result)
        
    finally:
        browser_manager.close()
```

## 💾 Memory and Document Management

### Memory Configuration with Builder

```python
import asyncio
from saplings import AgentBuilder

async def memory_example():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_memory_path("./knowledge_base") \
        .with_retrieval_max_documents(15) \
        .with_retrieval_entropy_threshold(0.1) \
        .build()
    
    # Add documents to memory
    agent.add_document(
        content="Python is a high-level programming language known for its simplicity and readability.",
        metadata={"topic": "programming", "language": "python"}
    )
    
    # Add documents from files
    agent.add_documents_from_directory("./docs", extension=".md")
    
    # Use memory in tasks
    result = await agent.run(
        "Based on the documents in your memory, explain the relationship between Python and machine learning"
    )
    
    print(result)

asyncio.run(memory_example())
```

## 🎯 Advanced Features

### Graph-Aligned Sparse Attention (GASA)

```python
from saplings import AgentBuilder

# Configure GASA for better performance
agent = AgentBuilder.for_openai("gpt-4o") \
    .with_gasa_enabled(True) \
    .with_gasa_max_hops(3) \
    .with_gasa_strategy("binary") \
    .with_gasa_fallback("prompt_composer") \
    .with_gasa_shadow_model_enabled(True) \
    .with_gasa_shadow_model_name("Qwen/Qwen3-0.6B") \
    .build()
```

### Self-Healing

```python
from saplings import AgentBuilder

agent = AgentBuilder.for_openai("gpt-4o") \
    .with_self_healing_enabled(True) \
    .with_self_healing_max_retries(3) \
    .build()

# The agent will automatically retry failed operations
result = await agent.run("Complex task that might fail initially")
```

### Planning and Execution

```python
from saplings import AgentBuilder

agent = AgentBuilder.for_openai("gpt-4o") \
    .with_planner_budget_strategy("dynamic") \
    .with_planner_total_budget(5.0) \
    .with_planner_allow_budget_overflow(True) \
    .build()

# Create explicit plans
plan = await agent.plan(
    task="Research renewable energy trends and create a comprehensive report",
    context=[]
)

# Execute with plan
result = await agent.execute_plan(plan=plan, use_tools=True)
```

### Multi-Modal Processing

```python
from saplings import AgentBuilder

agent = AgentBuilder.for_openai("gpt-4o") \
    .with_supported_modalities(["text", "image", "audio"]) \
    .build()

# Process different input types
result = await agent.run(
    task="Describe this image and transcribe any text in it",
    input_modalities=["image"],
    context=[{"type": "image", "path": "./image.jpg"}]
)
```

## 📊 Real-World Examples

### Research Assistant

```python
import asyncio
from saplings import AgentBuilder

async def research_assistant():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tools([
            "DuckDuckGoSearchTool",
            "WikipediaSearchTool",
            "PythonInterpreterTool"
        ]) \
        .with_gasa_enabled(True) \
        .with_monitoring_enabled(True) \
        .with_memory_path("./research_memory") \
        .build()
    
    result = await agent.run("""
    Research the current state of quantum computing in 2024:
    1. Find the latest breakthroughs and developments
    2. Identify the major players and companies
    3. Create visualizations showing progress over time
    4. Write a comprehensive summary report
    """)
    
    print(result)

asyncio.run(research_assistant())
```

### Data Science Pipeline

```python
import asyncio
from saplings import AgentBuilder

async def data_science_pipeline():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tools(["PythonInterpreterTool"]) \
        .with_memory_path("./data_science_memory") \
        .with_monitoring_enabled(True) \
        .with_allowed_imports([
            "pandas", "numpy", "matplotlib", "seaborn", 
            "scikit-learn", "scipy", "plotly"
        ]) \
        .build()
    
    # Load datasets
    agent.add_documents_from_directory("./datasets", extension=".csv")
    
    result = await agent.run("""
    Perform a complete data science analysis:
    1. Load and explore the datasets in memory
    2. Perform data cleaning and preprocessing
    3. Conduct exploratory data analysis with visualizations
    4. Build and evaluate machine learning models
    5. Generate insights and recommendations
    """)
    
    print(result)

asyncio.run(data_science_pipeline())
```

### Content Creation System

```python
import asyncio
from saplings import AgentBuilder

async def content_creator():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_supported_modalities(["text", "image"]) \
        .with_tools([
            "DuckDuckGoSearchTool",
            "WikipediaSearchTool",
            "PythonInterpreterTool"
        ]) \
        .with_monitoring_enabled(True) \
        .build()
    
    result = await agent.run(
        task="""
        Create a comprehensive blog post about sustainable energy solutions:
        1. Research current renewable energy technologies
        2. Create informative charts and infographics
        3. Write engaging content with proper SEO optimization
        4. Generate social media snippets
        """,
        output_modalities=["text", "image"]
    )
    
    print(result)

asyncio.run(content_creator())
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

## 📚 Documentation

### Complete Documentation

- **[Getting Started Guide](GETTING_STARTED.md)** - Step-by-step tutorial for beginners
- **[Configuration Guide](CONFIGURATION_GUIDE.md)** - Comprehensive configuration documentation
- **[API Reference](API_REFERENCE.md)** - Detailed API documentation
- **[Examples](EXAMPLES.md)** - Practical examples and use cases
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Best practices and advanced topics
- **[GASA Guide](GASA_GUIDE.md)** - Complete Graph-Aligned Sparse Attention documentation
- **[Service Builders Guide](SERVICE_BUILDERS_GUIDE.md)** - Complete service builder documentation
- **[Performance Optimization](PERFORMANCE_OPTIMIZATION_GUIDE.md)** - Optimization strategies

### Quick Reference

#### Core Builder Classes

- **`AgentBuilder`**: Primary builder for creating Agent instances
- **`