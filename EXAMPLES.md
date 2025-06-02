# Saplings Examples

This document provides practical examples demonstrating how to use the Saplings AI agent framework for various real-world applications. All examples use the recommended builder pattern as the primary interface.

## Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Configuration Examples](#configuration-examples)
- [Tool Integration Examples](#tool-integration-examples)
- [Memory and Retrieval Examples](#memory-and-retrieval-examples)
- [Advanced Feature Examples](#advanced-feature-examples)
- [Real-World Applications](#real-world-applications)

## Quick Start Examples

### Basic Agent Usage

```python
import asyncio
from saplings import AgentBuilder

async def basic_example():
    # Create an agent using the builder pattern (recommended)
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_api_key("your-openai-api-key") \
        .build()
    
    # Run a simple task
    result = await agent.run("What is the capital of France?")
    print(result)

# Run the example
asyncio.run(basic_example())
```

### Synchronous Usage

```python
from saplings import AgentBuilder

# For environments that don't support async/await
agent = AgentBuilder.for_openai("gpt-4o").build()
result = agent.run_sync("Explain quantum computing in simple terms")
print(result)
```

### Using Different Providers

```python
import asyncio
from saplings import AgentBuilder

async def provider_examples():
    # OpenAI
    openai_agent = AgentBuilder.for_openai("gpt-4o") \
        .with_api_key("your-key") \
        .build()
    
    # Anthropic
    anthropic_agent = AgentBuilder.for_anthropic("claude-3-opus") \
        .with_api_key("your-key") \
        .build()
    
    # Local vLLM
    vllm_agent = AgentBuilder.for_vllm("Qwen/Qwen3-7B-Instruct") \
        .build()
    
    # Run tasks with different providers
    tasks = [
        openai_agent.run("Explain machine learning"),
        anthropic_agent.run("Write a poem about AI"),
        vllm_agent.run("Code a simple calculator")
    ]
    
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"Provider {i+1} result: {result}")

asyncio.run(provider_examples())
```

## Configuration Examples

### Builder Pattern Configuration

```python
from saplings import AgentBuilder

# Complex configuration using builder pattern
agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_api_key("your-key") \
    .with_memory_path("./my_agent_memory") \
    .with_output_dir("./my_outputs") \
    .with_gasa_enabled(True) \
    .with_gasa_max_hops(3) \
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
        "top_p": 0.9
    }) \
    .build()
```

### Preset Configuration Patterns

```python
from saplings import AgentBuilder

# Minimal configuration for simple tasks
minimal_agent = AgentBuilder.minimal("openai", "gpt-4o") \
    .with_api_key("your-key") \
    .build()

# Standard configuration for most use cases
standard_agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_api_key("your-key") \
    .build()

# Full-featured configuration for complex tasks
full_agent = AgentBuilder.full_featured("openai", "gpt-4o") \
    .with_api_key("your-key") \
    .build()
```

### Environment-Based Configuration

```python
import os
from saplings import AgentBuilder

def create_agent_from_env():
    """Create agent configuration from environment variables."""
    provider = os.getenv("SAPLINGS_PROVIDER", "openai")
    model_name = os.getenv("SAPLINGS_MODEL", "gpt-4o")
    
    builder = AgentBuilder()
    
    if provider == "openai":
        builder = AgentBuilder.for_openai(model_name)
    elif provider == "anthropic":
        builder = AgentBuilder.for_anthropic(model_name)
    elif provider == "vllm":
        builder = AgentBuilder.for_vllm(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Configure from environment
    return builder \
        .with_monitoring_enabled(os.getenv("SAPLINGS_MONITORING", "true").lower() == "true") \
        .with_memory_path(os.getenv("SAPLINGS_MEMORY_PATH", "./agent_memory")) \
        .build()

# Usage
agent = create_agent_from_env()
```

## Tool Integration Examples

### Built-in Tools

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
    Research the latest Python 3.12 features, then write a Python script 
    that demonstrates at least 3 new features with examples and explanations.
    """)
    
    print(result)

asyncio.run(tool_example())
```

### Custom Tools

```python
import asyncio
from saplings import AgentBuilder
from saplings.api.tools import tool
import requests

@tool(name="weather", description="Get weather information for a city")
def get_weather(city: str, api_key: str = None) -> str:
    """
    Get current weather for a city.
    
    Args:
        city: Name of the city
        api_key: OpenWeatherMap API key
    
    Returns:
        Weather information as a string
    """
    if not api_key:
        return "Weather API key not provided"
    
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"Weather in {city}: {temp}°C, {desc}"
        else:
            return f"Error getting weather: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"

async def custom_tool_example():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_custom_tools([get_weather]) \
        .build()
    
    # Use the tool
    result = await agent.run(
        "Get the weather for Paris and suggest appropriate clothing"
    )
    print(result)

asyncio.run(custom_tool_example())
```

### Tool Factory (Dynamic Tools)

```python
import asyncio
from saplings import AgentBuilder

async def dynamic_tool_example():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tool_factory_enabled(True) \
        .with_tool_factory_sandbox_enabled(True) \
        .build()
    
    # Agent can create tools dynamically
    result = await agent.run("""
    I need to analyze CSV data. Create a tool that can:
    1. Load a CSV file
    2. Show basic statistics
    3. Create simple visualizations
    
    Then use it to analyze a sample dataset.
    """)
    
    print(result)

asyncio.run(dynamic_tool_example())
```

### Browser Automation

```python
import asyncio
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
                "ScreenshotTool",
                "ExtractTextTool"
            ]) \
            .build()
        
        result = await agent.run("""
        Go to https://example.com, take a screenshot, 
        extract the main text content, and summarize what the page is about.
        """)
        
        print(result)
        
    finally:
        browser_manager.close()

asyncio.run(browser_example())
```

## Memory and Retrieval Examples

### Document Management

```python
import asyncio
from saplings import AgentBuilder

async def memory_example():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_memory_path("./knowledge_base") \
        .build()
    
    # Add documents to memory
    agent.add_document(
        content="Python is a high-level programming language known for its simplicity and readability.",
        metadata={"topic": "programming", "language": "python"}
    )
    
    agent.add_document(
        content="Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        metadata={"topic": "ai", "subtopic": "machine_learning"}
    )
    
    # Add documents from files
    agent.add_documents_from_directory("./docs", extension=".md")
    
    # Add document from URL
    await agent.add_document_from_url("https://example.com/article")
    
    # Use memory in tasks
    result = await agent.run(
        "Based on the documents in your memory, explain the relationship between Python and machine learning"
    )
    
    print(result)

asyncio.run(memory_example())
```

### Advanced Retrieval

```python
import asyncio
from saplings import AgentBuilder

async def retrieval_example():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_retrieval_max_documents(10) \
        .with_retrieval_entropy_threshold(0.1) \
        .build()
    
    # Add some documents
    documents = [
        "Neural networks are computational models inspired by biological neural networks.",
        "Deep learning uses multiple layers to progressively extract higher-level features.",
        "Convolutional neural networks are particularly effective for image processing.",
        "Transformer models have revolutionized natural language processing.",
        "BERT and GPT are examples of transformer-based language models."
    ]
    
    for doc in documents:
        agent.add_document(doc, metadata={"domain": "ai"})
    
    # Retrieve relevant documents
    relevant_docs = await agent.retrieve(
        query="transformer models for language",
        limit=3,
        fast_mode=False
    )
    
    print("Retrieved documents:")
    for doc in relevant_docs:
        print(f"- {doc['content']}")
    
    # Use retrieved context
    result = await agent.run(
        task="Explain transformer models",
        context=relevant_docs,
        skip_retrieval=False  # Allow additional retrieval
    )
    
    print(f"\nResponse: {result}")

asyncio.run(retrieval_example())
```

## Advanced Feature Examples

### Graph-Aligned Sparse Attention (GASA)

```python
import asyncio
from saplings import AgentBuilder

async def gasa_example():
    # Configure GASA for better performance
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_gasa_enabled(True) \
        .with_gasa_max_hops(3) \
        .with_gasa_strategy("binary") \
        .with_gasa_fallback("prompt_composer") \
        .with_gasa_shadow_model_enabled(True) \
        .with_gasa_shadow_model_name("Qwen/Qwen3-0.6B") \
        .with_gasa_prompt_composer_enabled(True) \
        .build()
    
    # Add a large document to test GASA
    long_document = """
    [Imagine a very long document with multiple sections about AI, machine learning,
    deep learning, neural networks, transformers, etc. GASA will help focus on
    relevant parts when answering questions.]
    """
    
    agent.add_document(long_document, metadata={"type": "comprehensive_guide"})
    
    # GASA will help focus on relevant parts
    result = await agent.run(
        "What are the key advantages of transformer architectures over RNNs?"
    )
    
    print(result)

asyncio.run(gasa_example())
```

### Self-Healing

```python
import asyncio
from saplings import AgentBuilder

async def self_healing_example():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_self_healing_enabled(True) \
        .with_self_healing_max_retries(3) \
        .build()
    
    # The agent will automatically retry and learn from failures
    result = await agent.run("""
    Write a Python script that connects to a database,
    processes data, and generates a report. Handle any potential errors gracefully.
    """)
    
    print(result)
    
    # Trigger self-improvement
    improvement_result = await agent.self_improve()
    print(f"Self-improvement result: {improvement_result}")

asyncio.run(self_healing_example())
```

### Planning and Execution

```python
import asyncio
from saplings import AgentBuilder

async def planning_example():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_planner_budget_strategy("dynamic") \
        .with_planner_total_budget(5.0) \
        .with_planner_allow_budget_overflow(True) \
        .build()
    
    # Create a plan for a complex task
    plan = await agent.plan(
        task="Research renewable energy trends and create a comprehensive report with visualizations",
        context=[]
    )
    
    print("Generated plan:")
    for i, step in enumerate(plan, 1):
        print(f"{i}. {step.task_description} (Cost: ${step.estimated_cost:.2f})")
    
    # Execute the plan
    result = await agent.execute_plan(
        plan=plan,
        context=[],
        use_tools=True
    )
    
    print(f"\nExecution result: {result}")

asyncio.run(planning_example())
```

### Monitoring and Validation

```python
import asyncio
from saplings import AgentBuilder
from saplings.api.validator import CodeValidator, FactualValidator

async def monitoring_example():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_monitoring_enabled(True) \
        .with_executor_validation_type("judge") \
        .with_validators([
            CodeValidator(),
            FactualValidator()
        ]) \
        .build()
    
    # Run a task with full monitoring
    result = await agent.run(
        "Write a Python function to implement quicksort algorithm",
        save_results=True
    )
    
    # Validate the output
    validation_result = await agent.judge_output(
        input_data="Write a Python function to implement quicksort algorithm",
        output_data=result,
        judgment_type="code_quality"
    )
    
    print(f"Result: {result}")
    print(f"Validation: {validation_result}")
    
    # Get monitoring data
    traces = agent.get_execution_traces()
    metrics = agent.get_performance_metrics()
    
    print(f"Execution traces: {len(traces)}")
    print(f"Performance metrics: {metrics}")

asyncio.run(monitoring_example())
```

## Real-World Applications

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
    3. Analyze market trends and investment patterns
    4. Create visualizations showing progress over time
    5. Write a comprehensive summary report
    """)
    
    print(result)

asyncio.run(research_assistant())
```

### Code Analysis and Documentation

```python
import asyncio
from saplings import AgentBuilder

async def code_analyzer():
    agent = AgentBuilder.for_anthropic("claude-3-opus") \
        .with_tools(["PythonInterpreterTool"]) \
        .with_tool_factory_enabled(True) \
        .with_tool_factory_sandbox_enabled(True) \
        .with_memory_path("./code_analysis") \
        .build()
    
    # Add codebase to memory
    agent.add_documents_from_directory("./src", extension=".py")
    agent.add_documents_from_directory("./tests", extension=".py")
    
    result = await agent.run("""
    Analyze the codebase in memory and provide:
    1. Code quality assessment
    2. Potential bugs and security issues
    3. Performance optimization suggestions
    4. Missing documentation and tests
    5. Refactoring recommendations
    6. Generate missing unit tests
    7. Create API documentation
    """)
    
    print(result)

asyncio.run(code_analyzer())
```

### Data Science Pipeline

```python
import asyncio
from saplings import AgentBuilder

async def data_science_pipeline():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tools([
            "PythonInterpreterTool",
            "DuckDuckGoSearchTool"
        ]) \
        .with_memory_path("./data_science_memory") \
        .with_monitoring_enabled(True) \
        .with_allowed_imports([
            "pandas", "numpy", "matplotlib", "seaborn", 
            "scikit-learn", "scipy", "plotly"
        ]) \
        .build()
    
    # Load datasets
    agent.add_documents_from_directory("./datasets", extension=".csv")
    agent.add_documents_from_directory("./data_docs", extension=".md")
    
    result = await agent.run("""
    Perform a complete data science analysis:
    1. Load and explore the datasets in memory
    2. Perform data cleaning and preprocessing
    3. Conduct exploratory data analysis with visualizations
    4. Build and evaluate machine learning models
    5. Generate insights and recommendations
    6. Create a presentation-ready report
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
            "VisitWebpageTool",
            "PythonInterpreterTool"
        ]) \
        .with_monitoring_enabled(True) \
        .build()
    
    result = await agent.run(
        task="""
        Create a comprehensive blog post about sustainable energy solutions:
        1. Research current renewable energy technologies
        2. Analyze market trends and statistics
        3. Create informative charts and infographics
        4. Write engaging content with proper SEO optimization
        5. Generate social media snippets
        6. Create a content calendar for follow-up posts
        """,
        output_modalities=["text", "image"]
    )
    
    print(result)

asyncio.run(content_creator())
```

### Multi-Modal AI Assistant

```python
import asyncio
from saplings import AgentBuilder

async def multimodal_assistant():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_supported_modalities(["text", "image", "audio"]) \
        .with_tools([
            "PythonInterpreterTool",
            "VisitWebpageTool",
            "SpeechToTextTool"
        ]) \
        .build()
    
    # Process different types of inputs
    result = await agent.run(
        task="Analyze this image, transcribe any audio content, and create a summary report",
        input_modalities=["image", "audio"],
        context=[
            {"type": "image", "path": "./presentation_slide.jpg"},
            {"type": "audio", "path": "./meeting_recording.wav"}
        ]
    )
    
    print(result)

asyncio.run(multimodal_assistant())
```

### Customer Support Agent

```python
import asyncio
from saplings import AgentBuilder

async def customer_support():
    agent = AgentBuilder.for_anthropic("claude-3-sonnet") \
        .with_tools([
            "DuckDuckGoSearchTool",
            "UserInputTool",
            "FinalAnswerTool"
        ]) \
        .with_self_healing_enabled(True) \
        .with_memory_path("./support_knowledge") \
        .build()
    
    # Load knowledge base
    agent.add_documents_from_directory("./support_docs", extension=".md")
    agent.add_documents_from_directory("./faq", extension=".txt")
    
    # Interactive support session
    result = await agent.run("""
    Act as a customer support agent. Help users with their questions by:
    1. Understanding their problem clearly
    2. Searching the knowledge base for relevant information
    3. Providing step-by-step solutions
    4. Following up to ensure the issue is resolved
    5. Escalating to human agents when necessary
    """)
    
    print(result)

asyncio.run(customer_support())
```

### Automated Testing Assistant

```python
import asyncio
from saplings import AgentBuilder

async def testing_assistant():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tools(["PythonInterpreterTool"]) \
        .with_tool_factory_enabled(True) \
        .with_memory_path("./test_knowledge") \
        .with_executor_validation_type("execution") \
        .build()
    
    # Add project documentation
    agent.add_documents_from_directory("./docs", extension=".md")
    agent.add_documents_from_directory("./src", extension=".py")
    
    result = await agent.run("""
    Analyze the codebase and create comprehensive tests:
    1. Identify all functions and classes that need testing
    2. Generate unit tests with good coverage
    3. Create integration tests for main workflows
    4. Add performance benchmarks
    5. Generate test documentation
    6. Create CI/CD pipeline configuration
    """)
    
    print(result)

asyncio.run(testing_assistant())
```

### Convenience Constructor Examples

For simple use cases where you don't need the full builder configuration, you can use the direct constructor:

```python
from saplings import Agent

# Simple case - direct instantiation
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-api-key"
)

# With basic configuration
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    memory_path="./simple_memory",
    tools=["PythonInterpreterTool"]
)
```

However, the builder pattern is recommended for most use cases as it provides:
- Better discoverability of configuration options
- Type safety and validation
- More maintainable code
- Better error messages

These examples demonstrate the versatility and power of the Saplings framework using the recommended builder pattern. You can adapt and combine these patterns to build sophisticated AI applications for your specific needs.