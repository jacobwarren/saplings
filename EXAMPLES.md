# Saplings Examples

This document provides practical examples demonstrating how to use the Saplings AI agent framework for various real-world applications.

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
from saplings import Agent

async def basic_example():
    # Create a simple agent
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        api_key="your-openai-api-key"
    )
    
    # Run a simple task
    result = await agent.run("What is the capital of France?")
    print(result)

# Run the example
asyncio.run(basic_example())
```

### Synchronous Usage

```python
from saplings import Agent

# For environments that don't support async/await
agent = Agent(provider="openai", model_name="gpt-4o")
result = agent.run_sync("Explain quantum computing in simple terms")
print(result)
```

### Using Different Providers

```python
import asyncio
from saplings import Agent, AgentConfig

async def provider_examples():
    # OpenAI
    openai_agent = Agent.from_config(
        AgentConfig.for_openai("gpt-4o", api_key="your-key")
    )
    
    # Anthropic
    anthropic_agent = Agent.from_config(
        AgentConfig.for_anthropic("claude-3-opus", api_key="your-key")
    )
    
    # Local vLLM
    vllm_agent = Agent.from_config(
        AgentConfig.for_vllm("Qwen/Qwen3-7B-Instruct")
    )
    
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

### Custom Configuration Object

```python
from saplings import Agent, AgentConfig

# Detailed configuration
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-key",
    
    # GASA settings
    enable_gasa=True,
    gasa_max_hops=3,
    gasa_strategy="binary",
    gasa_fallback="prompt_composer",
    gasa_shadow_model=True,
    gasa_shadow_model_name="Qwen/Qwen3-0.6B",
    
    # Memory and retrieval
    memory_path="./custom_memory",
    retrieval_max_documents=15,
    retrieval_entropy_threshold=0.15,
    
    # Planning
    planner_budget_strategy="dynamic",
    planner_total_budget=3.0,
    planner_allow_budget_overflow=True,
    
    # Advanced features
    enable_self_healing=True,
    enable_tool_factory=True,
    enable_monitoring=True,
    
    # Multi-modal support
    supported_modalities=["text", "image", "audio"]
)

agent = Agent(config=config)
```

### Configuration Presets

```python
from saplings import Agent, AgentConfig

# Minimal configuration for simple tasks
minimal_agent = Agent.from_config(
    AgentConfig.minimal("openai", "gpt-4o", api_key="your-key")
)

# Standard configuration for most use cases
standard_agent = Agent.from_config(
    AgentConfig.standard("openai", "gpt-4o", api_key="your-key")
)

# Full-featured configuration for complex tasks
full_agent = Agent.from_config(
    AgentConfig.full_featured("openai", "gpt-4o", api_key="your-key")
)
```

## Tool Integration Examples

### Built-in Tools

```python
import asyncio
from saplings import Agent
from saplings.api.tools import (
    PythonInterpreterTool,
    DuckDuckGoSearchTool,
    WikipediaSearchTool,
    VisitWebpageTool
)

async def tool_example():
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
from saplings import Agent
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
    agent = Agent(provider="openai", model_name="gpt-4o")
    
    # Register the custom tool
    agent.register_tool(get_weather)
    
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
from saplings import Agent

async def dynamic_tool_example():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        enable_tool_factory=True,
        tool_factory_sandbox_enabled=True
    )
    
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
from saplings import Agent
from saplings.api.browser_tools import BrowserManager

async def browser_example():
    # Initialize browser manager
    browser_manager = BrowserManager(headless=False)
    browser_manager.initialize()
    
    try:
        agent = Agent(
            provider="openai",
            model_name="gpt-4o",
            tools=[
                "GoToTool",
                "ClickTool",
                "TypeTool",
                "ScreenshotTool",
                "ExtractTextTool"
            ]
        )
        
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
from saplings import Agent

async def memory_example():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        memory_path="./knowledge_base"
    )
    
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
from saplings import Agent

async def retrieval_example():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        retrieval_max_documents=10,
        retrieval_entropy_threshold=0.1
    )
    
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
from saplings import Agent, AgentConfig

async def gasa_example():
    # Configure GASA for better performance
    config = AgentConfig.for_openai(
        "gpt-4o",
        enable_gasa=True,
        gasa_max_hops=3,
        gasa_strategy="binary",
        gasa_fallback="prompt_composer",
        gasa_shadow_model=True,
        gasa_shadow_model_name="Qwen/Qwen3-0.6B",
        gasa_prompt_composer=True
    )
    
    agent = Agent(config=config)
    
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
from saplings import Agent

async def self_healing_example():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        enable_self_healing=True,
        self_healing_max_retries=3
    )
    
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
from saplings import Agent

async def planning_example():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        planner_budget_strategy="dynamic",
        planner_total_budget=5.0,
        planner_allow_budget_overflow=True
    )
    
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
from saplings import Agent
from saplings.api.validator import CodeValidator, FactualValidator

async def monitoring_example():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        enable_monitoring=True,
        executor_validation_type="judge",
        validators=[
            CodeValidator(),
            FactualValidator()
        ]
    )
    
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
from saplings import Agent
from saplings.api.tools import DuckDuckGoSearchTool, WikipediaSearchTool, PythonInterpreterTool

async def research_assistant():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        tools=[
            DuckDuckGoSearchTool(),
            WikipediaSearchTool(),
            PythonInterpreterTool()
        ],
        enable_gasa=True,
        enable_monitoring=True,
        memory_path="./research_memory"
    )
    
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
from saplings import Agent
from saplings.api.tools import PythonInterpreterTool

async def code_analyzer():
    agent = Agent(
        provider="anthropic",
        model_name="claude-3-opus",
        tools=[PythonInterpreterTool()],
        enable_tool_factory=True,
        tool_factory_sandbox_enabled=True,
        memory_path="./code_analysis"
    )
    
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
import pandas as pd

async def data_science_pipeline():
    agent = AgentBuilder() \
        .with_provider("openai") \
        .with_model_name("gpt-4o") \
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
from saplings import Agent

async def content_creator():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        supported_modalities=["text", "image"],
        tools=[
            "DuckDuckGoSearchTool",
            "WikipediaSearchTool",
            "VisitWebpageTool",
            "PythonInterpreterTool"
        ],
        enable_monitoring=True
    )
    
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
from saplings import Agent

async def multimodal_assistant():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        supported_modalities=["text", "image", "audio"],
        tools=[
            "PythonInterpreterTool",
            "VisitWebpageTool",
            "SpeechToTextTool"
        ]
    )
    
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
from saplings import Agent

async def customer_support():
    agent = Agent(
        provider="anthropic",
        model_name="claude-3-sonnet",
        tools=[
            "DuckDuckGoSearchTool",
            "UserInputTool",
            "FinalAnswerTool"
        ],
        enable_self_healing=True,
        memory_path="./support_knowledge"
    )
    
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

These examples demonstrate the versatility and power of the Saplings framework. You can adapt and combine these patterns to build sophisticated AI applications for your specific needs.