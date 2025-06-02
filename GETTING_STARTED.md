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
from saplings import Agent

async def main():
    # Create a simple agent
    agent = Agent(
        provider="openai",  # or "anthropic", "vllm"
        model_name="gpt-4o",
        api_key="your-api-key-here"  # or set OPENAI_API_KEY environment variable
    )
    
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

### Option 2: Pass API Key Directly

```python
agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-api-key-here"
)
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
load_dotenv()

agent = Agent(provider="openai", model_name="gpt-4o")
```

## Step 4: Using Different Providers

### OpenAI

```python
from saplings import Agent, AgentConfig

agent = Agent.from_config(
    AgentConfig.for_openai("gpt-4o", api_key="your-key")
)
```

### Anthropic

```python
agent = Agent.from_config(
    AgentConfig.for_anthropic("claude-3-opus", api_key="your-key")
)
```

### Local vLLM Server

```python
agent = Agent.from_config(
    AgentConfig.for_vllm("Qwen/Qwen3-7B-Instruct")
)
```

## Step 5: Adding Tools

Tools give your agent superpowers! Here's how to add some built-in tools:

```python
import asyncio
from saplings import Agent
from saplings.api.tools import (
    PythonInterpreterTool,
    DuckDuckGoSearchTool,
    WikipediaSearchTool
)

async def main():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        tools=[
            PythonInterpreterTool(),      # Execute Python code
            DuckDuckGoSearchTool(),       # Search the web
            WikipediaSearchTool()         # Search Wikipedia
        ]
    )
    
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
from saplings import Agent

async def main():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        memory_path="./my_agent_memory"  # Persistent storage
    )
    
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

Customize your agent's behavior:

```python
from saplings import Agent, AgentConfig

# Create a custom configuration
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    
    # Memory settings
    memory_path="./knowledge_base",
    
    # Advanced features
    enable_gasa=True,           # Graph-Aligned Sparse Attention
    enable_monitoring=True,     # Performance monitoring
    enable_self_healing=True,   # Automatic error recovery
    
    # Planning settings
    planner_budget_strategy="dynamic",
    planner_total_budget=5.0,   # $5 USD budget
    
    # Model parameters
    temperature=0.7,
    max_tokens=2048
)

agent = Agent(config=config)
```

## Step 8: Synchronous Usage

If you're working in a non-async environment:

```python
from saplings import Agent

# Create agent
agent = Agent(provider="openai", model_name="gpt-4o")

# Use synchronous wrapper
result = agent.run_sync("Explain quantum computing in simple terms")
print(result)
```

## Step 9: Creating Custom Tools

Make your own tools:

```python
from saplings import Agent
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

# Create agent and register tool
agent = Agent(provider="openai", model_name="gpt-4o")
agent.register_tool(calculate)

# Use the tool
result = agent.run_sync("What's 15 * 7 + 23?")
print(result)
```

## Step 10: Error Handling

Handle errors gracefully:

```python
import asyncio
from saplings import Agent
from saplings.core.exceptions import ConfigurationError, ModelError

async def main():
    try:
        agent = Agent(provider="openai", model_name="gpt-4o")
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
from saplings import Agent
from saplings.api.tools import DuckDuckGoSearchTool, WikipediaSearchTool

async def research_assistant():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        tools=[DuckDuckGoSearchTool(), WikipediaSearchTool()]
    )
    
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
from saplings import Agent
from saplings.api.tools import PythonInterpreterTool

async def code_helper():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        tools=[PythonInterpreterTool()]
    )
    
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
from saplings import Agent

async def document_qa():
    agent = Agent(
        provider="openai",
        model_name="gpt-4o",
        memory_path="./documents"
    )
    
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
from saplings import AgentConfig

config = AgentConfig.for_openai("gpt-4o", enable_gasa=True)
agent = Agent(config=config)
```

## Getting Help

- **Documentation**: Check the full [API Reference](API_REFERENCE.md)
- **Examples**: See [Examples](EXAMPLES.md) for more use cases
- **Issues**: Report bugs on [GitHub Issues](https://github.com/jacobwarren/saplings/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/jacobwarren/saplings/discussions)

Happy building with Saplings! 🌱