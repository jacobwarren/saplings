# Getting Started with Saplings

Welcome to Saplings! This guide will help you get up and running with the AI agent framework in just a few minutes.

## Prerequisites

- Python 3.10 or higher
- An API key for OpenAI, Anthropic, or access to a local LLM server

## Step 1: Installation

### Basic Installation

```bash
pip install saplings
```

### Full Installation (Recommended)

For the best experience with all features:

```bash
pip install saplings[full]
```

### Specific Feature Installation

Choose only the features you need:

```bash
# For transformers and tools
pip install saplings[transformers,tools]

# For browser automation
pip install saplings[browser]

# For enhanced retrieval
pip install saplings[retrieval,faiss]
```

## Step 2: Your First Agent

Create a file called `first_agent.py`:

```python
import asyncio
from saplings import AgentBuilder

async def main():
    # Create an agent using the builder API (recommended)
    agent = AgentBuilder() \
        .with_provider("openai") \
        .with_model_name("gpt-4o") \
        .with_api_key("your-api-key-here") \
        .build()
    
    # Ask a simple question
    result = await agent.run("What is the capital of France?")
    print(result)

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```

Run your first agent:

```bash
python first_agent.py
```

## Step 3: Setting Up API Keys

### Option 1: Environment Variables (Recommended)

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Windows (Command Prompt)
set OPENAI_API_KEY=your-openai-api-key
```

When using environment variables, you can omit the API key from your code:

```python
from saplings import AgentBuilder

agent = AgentBuilder.for_openai("gpt-4o").build()
```

### Option 2: Pass API Key Directly

```python
agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_api_key("your-api-key-here") \
    .build()
```

### Option 3: Using Configuration Files

Create a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Then load it in your script:

```python
from dotenv import load_dotenv
from saplings import AgentBuilder

load_dotenv()

agent = AgentBuilder.for_openai("gpt-4o").build()
```

## Step 4: Using Different Providers

### OpenAI (Recommended for Beginners)

```python
from saplings import AgentBuilder

# Using preset factory method
agent = AgentBuilder.for_openai("gpt-4o") \
    .with_api_key("your-key") \
    .build()
```

### Anthropic

```python
agent = AgentBuilder.for_anthropic("claude-3-opus") \
    .with_api_key("your-key") \
    .build()
```

### Local vLLM Server

```python
agent = AgentBuilder.for_vllm("Qwen/Qwen3-7B-Instruct") \
    .build()
```

### Manual Configuration

```python
agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_api_key("your-key") \
    .build()
```

## Step 5: Adding Tools

Tools give your agent superpowers! Here's how to add some built-in tools:

```python
import asyncio
from saplings import AgentBuilder

async def main():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tools([
            "PythonInterpreterTool",      # Execute Python code
            "DuckDuckGoSearchTool",       # Search the web
            "WikipediaSearchTool"         # Search Wikipedia
        ]) \
        .build()
    
    # Now your agent can code and search!
    result = await agent.run("""
    Search for information about Python programming, 
    then write a simple Python script that demonstrates 
    a basic concept you found.
    """)
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 6: Adding Memory

Give your agent persistent memory:

```python
import asyncio
from saplings import AgentBuilder

async def main():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_memory_path("./my_agent_memory") \
        .build()
    
    # Add some knowledge to memory
    agent.add_document(
        content="Saplings is an AI agent framework that supports multiple LLM providers.",
        metadata={"topic": "saplings", "type": "documentation"}
    )
    
    # The agent can now reference this information
    result = await agent.run("What do you know about Saplings?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 7: Configuration Options

The builder API makes all configuration options discoverable:

```python
from saplings import AgentBuilder

# Comprehensive configuration using the builder
agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_api_key("your-key") \
    .with_memory_path("./knowledge_base") \
    .with_gasa_enabled(True) \
    .with_monitoring_enabled(True) \
    .with_self_healing_enabled(True) \
    .with_planner_budget_strategy("dynamic") \
    .with_planner_total_budget(5.0) \
    .with_temperature(0.7) \
    .with_max_tokens(2048) \
    .build()
```

### Using Preset Configurations

For common scenarios, use preset methods:

```python
from saplings import AgentBuilder

# Minimal configuration for simple tasks
agent = AgentBuilder.minimal("openai", "gpt-4o").build()

# Standard configuration for most use cases
agent = AgentBuilder.standard("openai", "gpt-4o").build()

# Full-featured configuration for complex tasks
agent = AgentBuilder.full_featured("openai", "gpt-4o").build()
```

## Step 8: Synchronous Usage

If you're working in a non-async environment:

```python
from saplings import AgentBuilder

# Create agent
agent = AgentBuilder.for_openai("gpt-4o").build()

# Use synchronous wrapper
result = agent.run_sync("Explain quantum computing in simple terms")
print(result)
```

## Step 9: Creating Custom Tools

Make your own tools:

```python
from saplings import AgentBuilder
from saplings.api.tools import tool

@tool(name="calculator", description="Performs basic math calculations")
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    
    Args:
        expression: Math expression like "2 + 3 * 4"
    
    Returns:
        The calculation result
    """
    try:
        # Safe evaluation for basic math
        result = eval(expression.replace('^', '**'))
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create agent with custom tool using builder
agent = AgentBuilder.for_openai("gpt-4o") \
    .with_custom_tools([calculate]) \
    .build()

# Use the tool
result = agent.run_sync("What's 15 * 7 + 23?")
print(result)
```

## Step 10: Error Handling

Handle errors gracefully:

```python
import asyncio
from saplings import AgentBuilder
from saplings.core.exceptions import ConfigurationError, ModelError

async def main():
    try:
        agent = AgentBuilder.for_openai("gpt-4o").build()
        result = await agent.run("Hello, world!")
        print(result)
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        print("Check your API key and provider settings")
        
    except ModelError as e:
        print(f"Model error: {e}")
        print("There might be an issue with the model or API")
        
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Quick Examples

### Research Assistant

```python
import asyncio
from saplings import AgentBuilder

async def research_assistant():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tools(["DuckDuckGoSearchTool", "WikipediaSearchTool"]) \
        .build()
    
    result = await agent.run("""
    Research the latest developments in renewable energy in 2024. 
    Provide a summary with key trends and major breakthroughs.
    """)
    
    print(result)

asyncio.run(research_assistant())
```

### Code Helper

```python
import asyncio
from saplings import AgentBuilder

async def code_helper():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tools(["PythonInterpreterTool"]) \
        .build()
    
    result = await agent.run("""
    Write a Python function to find the longest word in a sentence,
    then test it with a few examples.
    """)
    
    print(result)

asyncio.run(code_helper())
```

### Document Q&A

```python
import asyncio
from saplings import AgentBuilder

async def document_qa():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_memory_path("./documents") \
        .build()
    
    # Add some documents
    agent.add_document(
        "The Solar System has 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
        metadata={"topic": "astronomy"}
    )
    
    agent.add_document(
        "Python is a programming language created by Guido van Rossum in 1991.",
        metadata={"topic": "programming"}
    )
    
    # Ask questions about the documents
    result = await agent.run("How many planets are in our Solar System?")
    print(result)

asyncio.run(document_qa())
```

### Multi-Step Workflow

```python
import asyncio
from saplings import AgentBuilder

async def multi_step_workflow():
    agent = AgentBuilder.for_openai("gpt-4o") \
        .with_tools([
            "DuckDuckGoSearchTool",
            "PythonInterpreterTool"
        ]) \
        .with_planner_budget_strategy("dynamic") \
        .with_planner_total_budget(3.0) \
        .build()
    
    result = await agent.run("""
    Create a comprehensive analysis:
    1. Research current AI trends
    2. Create a Python script to visualize the data
    3. Generate a summary report with key insights
    """)
    
    print(result)

asyncio.run(multi_step_workflow())
```

## Alternative: Direct Agent Constructor

For simple use cases, you can also use the direct constructor (convenience method):

```python
from saplings import Agent

# Direct instantiation
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-api-key-here"
)

result = await agent.run("What is the capital of France?")
```

However, the builder pattern is recommended because it:
- Makes configuration options more discoverable
- Provides better error messages
- Offers type safety and validation
- Is more maintainable as the framework evolves

## Next Steps

Now that you have the basics down, you can:

1. **Explore Advanced Features**: Try GASA (Graph-Aligned Sparse Attention), self-healing, and monitoring
2. **Build Complex Applications**: Check out the [Examples](EXAMPLES.md) for real-world use cases
3. **Read the Full Documentation**: See [API Reference](API_REFERENCE.md) for detailed information
4. **Join the Community**: Contribute to the project or ask questions

## Common Issues

### "Module not found" errors

Make sure you installed the right extras:

```bash
pip install saplings[full]
```

### API key errors

Double-check your API key and make sure it's set correctly:

```python
import os
print(os.getenv("OPENAI_API_KEY"))  # Should print your key
```

### Memory path issues

Make sure the directory exists and is writable:

```python
import os
os.makedirs("./agent_memory", exist_ok=True)
```

### Performance issues

Try enabling GASA for better performance:

```python
agent = AgentBuilder.for_openai("gpt-4o") \
    .with_gasa_enabled(True) \
    .build()
```

### Builder method not found

Make sure you're using the correct builder method names:

```python
# Correct
agent = AgentBuilder().with_provider("openai").build()

# Incorrect
agent = AgentBuilder().provider("openai").build()  # Missing 'with_'
```

## Getting Help

- **Documentation**: Check the full [API Reference](API_REFERENCE.md)
- **Examples**: See [Examples](EXAMPLES.md) for more use cases
- **Issues**: Report bugs on [GitHub Issues](https://github.com/jacobwarren/saplings/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/jacobwarren/saplings/discussions)

## Key Takeaways

1. **Use the Builder Pattern**: `AgentBuilder` is the primary way to create agents
2. **Start with Presets**: Use `.for_openai()`, `.for_anthropic()`, etc. for quick setup
3. **Environment Variables**: Set API keys as environment variables for security
4. **Tools are Powerful**: Add tools to give your agent capabilities
5. **Memory Persists**: Use memory paths to give agents long-term knowledge
6. **Error Handling**: Always wrap agent calls in try-catch blocks

Happy building with Saplings! 🌱