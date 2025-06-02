# Saplings Developer Guide

This guide provides best practices, architectural patterns, and troubleshooting information for developing with the Saplings AI agent framework.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Best Practices](#best-practices)
- [Configuration Patterns](#configuration-patterns)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Testing Strategies](#testing-strategies)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Architecture Overview

### Core Components

```
┌─────────────────┐
│     Agent       │  ← Main interface
├─────────────────┤
│   AgentFacade   │  ← Service orchestration
├─────────────────┤
│    Services     │  ← Business logic
├─────────────────┤
│   Adapters      │  ← External integrations
└─────────────────┘
```

### Service Layer Architecture

```python
from saplings.api.services import (
    MemoryService,          # Document storage
    RetrievalService,       # Information retrieval
    PlannerService,         # Task planning
    ExecutionService,       # Task execution
    ValidationService,      # Output validation
    ToolService,           # Tool management
    MonitoringService,      # Performance tracking
    ModelService,          # LLM interactions
    ModalityService,       # Multi-modal processing
    SelfHealingService     # Error recovery
)
```

### Dependency Injection

Saplings uses a dependency injection container for clean separation of concerns:

```python
from saplings.api.di import container, configure_container

# Configure services
configure_container(config)

# Services are automatically injected
memory_service = container.resolve("MemoryService")
```

## Best Practices

### 1. Configuration Management

**Use factory methods for common scenarios:**

```python
from saplings import AgentConfig

# Good: Use presets for common scenarios
config = AgentConfig.for_openai("gpt-4o", api_key=api_key)

# Avoid: Manual configuration for standard use cases
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key=api_key,
    enable_gasa=True,
    gasa_shadow_model=True,
    # ... many more settings
)
```

**Validate configuration early:**

```python
config = AgentConfig.for_openai("gpt-4o")

# Validate before creating agent
validation_result = config.validate()
if not validation_result.is_valid:
    print(f"Configuration error: {validation_result.message}")
    for suggestion in validation_result.suggestions:
        print(f"Suggestion: {suggestion}")
    exit(1)

agent = Agent(config=config)
```

### 2. Resource Management

**Always use context managers for browser tools:**

```python
from saplings.api.browser_tools import BrowserManager

async def web_automation_task():
    browser_manager = BrowserManager(headless=True)
    
    try:
        browser_manager.initialize()
        agent = Agent(
            provider="openai",
            model_name="gpt-4o",
            tools=["GoToTool", "ClickTool", "ScreenshotTool"]
        )
        
        result = await agent.run("Navigate to example.com and take a screenshot")
        return result
        
    finally:
        browser_manager.close()
```

**Manage memory efficiently:**

```python
# Good: Clear memory when no longer needed
agent.clear_memory()

# Good: Use specific memory paths for different contexts
research_agent = Agent(memory_path="./research_memory")
coding_agent = Agent(memory_path="./coding_memory")
```

### 3. Error Handling

**Use specific exception handling:**

```python
from saplings.core.exceptions import (
    ConfigurationError,
    ModelError,
    ToolError,
    RetrievalError
)

try:
    result = await agent.run("Complex task")
except ConfigurationError as e:
    logger.error(f"Configuration issue: {e}")
    # Handle configuration problems
except ModelError as e:
    logger.error(f"Model error: {e}")
    # Handle model-related issues
except ToolError as e:
    logger.error(f"Tool execution failed: {e}")
    # Handle tool failures
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle other errors
```

### 4. Tool Development

**Follow the tool interface:**

```python
from saplings.api.tools import tool

@tool(
    name="file_processor",
    description="Process files with various operations"
)
def process_file(
    file_path: str,
    operation: str = "read",
    encoding: str = "utf-8"
) -> str:
    """
    Process a file with the specified operation.
    
    Args:
        file_path: Path to the file to process
        operation: Operation to perform ("read", "count_lines", "get_size")
        encoding: File encoding for text operations
        
    Returns:
        Result of the file operation
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If access is denied
    """
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if operation == "read":
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    elif operation == "count_lines":
        with open(file_path, 'r', encoding=encoding) as f:
            return str(len(f.readlines()))
    elif operation == "get_size":
        return str(os.path.getsize(file_path))
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

**Validate tool implementations:**

```python
from saplings.api.tools import validate_tool

# Validate before registration
try:
    validate_tool(process_file)
    agent.register_tool(process_file)
except ValueError as e:
    print(f"Tool validation failed: {e}")
```

## Configuration Patterns

### Environment-Based Configuration

```python
import os
from saplings import AgentConfig

def create_config_from_env():
    """Create configuration from environment variables."""
    provider = os.getenv("SAPLINGS_PROVIDER", "openai")
    model_name = os.getenv("SAPLINGS_MODEL", "gpt-4o")
    api_key = os.getenv("SAPLINGS_API_KEY")
    
    if provider == "openai":
        config = AgentConfig.for_openai(model_name, api_key=api_key)
    elif provider == "anthropic":
        config = AgentConfig.for_anthropic(model_name, api_key=api_key)
    elif provider == "vllm":
        config = AgentConfig.for_vllm(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Override with environment-specific settings
    config.enable_monitoring = os.getenv("SAPLINGS_MONITORING", "true").lower() == "true"
    config.memory_path = os.getenv("SAPLINGS_MEMORY_PATH", "./agent_memory")
    
    return config
```

### Configuration Profiles

```python
from saplings import AgentConfig

class ConfigProfiles:
    @staticmethod
    def development():
        """Development configuration with debugging enabled."""
        return AgentConfig.standard(
            "openai", "gpt-4o",
            enable_monitoring=True,
            enable_self_healing=False,  # Disable for debugging
            memory_path="./dev_memory",
            planner_budget_strategy="fixed",
            planner_total_budget=1.0
        )
    
    @staticmethod
    def production():
        """Production configuration optimized for reliability."""
        return AgentConfig.full_featured(
            "openai", "gpt-4o",
            enable_monitoring=True,
            enable_self_healing=True,
            memory_path="/var/lib/saplings/memory",
            planner_budget_strategy="dynamic",
            planner_total_budget=10.0
        )
    
    @staticmethod
    def testing():
        """Testing configuration with minimal features."""
        return AgentConfig.minimal(
            "mock", "test-model",
            enable_monitoring=False,
            memory_path=":memory:",  # In-memory storage
        )
```

### Dynamic Configuration

```python
from saplings import Agent, AgentBuilder

class AdaptiveAgent:
    def __init__(self, base_config):
        self.base_config = base_config
        self.performance_history = []
    
    async def create_agent_for_task(self, task_type: str):
        """Create optimized agent based on task type."""
        builder = AgentBuilder()
        
        if task_type == "research":
            return builder \
                .with_provider(self.base_config.provider) \
                .with_model_name(self.base_config.model_name) \
                .with_tools([
                    "DuckDuckGoSearchTool",
                    "WikipediaSearchTool",
                    "VisitWebpageTool"
                ]) \
                .with_retrieval_max_documents(20) \
                .build()
        
        elif task_type == "coding":
            return builder \
                .with_provider(self.base_config.provider) \
                .with_model_name(self.base_config.model_name) \
                .with_tools([
                    "PythonInterpreterTool"
                ]) \
                .with_tool_factory_enabled(True) \
                .with_executor_validation_type("execution") \
                .build()
        
        else:
            return Agent(config=self.base_config)
```

## Error Handling

### Graceful Degradation

```python
import asyncio
from saplings import Agent
from saplings.core.exceptions import ToolError, ModelError

class RobustAgent:
    def __init__(self, config):
        self.primary_agent = Agent(config)
        # Fallback with minimal configuration
        fallback_config = config.minimal(config.provider, config.model_name)
        self.fallback_agent = Agent(fallback_config)
    
    async def run_with_fallback(self, task: str, **kwargs):
        """Run task with automatic fallback on failure."""
        try:
            return await self.primary_agent.run(task, **kwargs)
        
        except ToolError as e:
            print(f"Tool error, retrying without tools: {e}")
            return await self.primary_agent.run(task, use_tools=False, **kwargs)
        
        except ModelError as e:
            print(f"Model error, using fallback agent: {e}")
            return await self.fallback_agent.run(task, **kwargs)
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            # Log error and return safe default
            return f"Error processing task: {str(e)}"
```

### Retry Strategies

```python
import asyncio
from typing import Any, Callable
import random

class RetryStrategy:
    @staticmethod
    async def exponential_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True
    ) -> Any:
        """Retry with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                delay = min(base_delay * (2 ** attempt), max_delay)
                if jitter:
                    delay *= (0.5 + random.random() * 0.5)
                
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)

# Usage
async def run_with_retry(agent, task):
    return await RetryStrategy.exponential_backoff(
        lambda: agent.run(task),
        max_retries=3,
        base_delay=1.0
    )
```

## Performance Optimization

### Memory Management

```python
from saplings import Agent
import gc

class OptimizedAgent:
    def __init__(self, config):
        self.config = config
        self.agent = None
    
    async def process_batch(self, tasks: list[str]):
        """Process multiple tasks efficiently."""
        results = []
        
        # Create agent once for batch
        self.agent = Agent(self.config)
        
        try:
            for i, task in enumerate(tasks):
                result = await self.agent.run(task)
                results.append(result)
                
                # Clear memory periodically
                if i % 10 == 0:
                    gc.collect()
                    
        finally:
            # Clean up
            if self.agent:
                self.agent.clear_memory()
                self.agent = None
            gc.collect()
        
        return results
```

### Caching Strategies

```python
import asyncio
from functools import wraps
import hashlib
import json

def cache_results(ttl_seconds: int = 3600):
    """Cache agent results for repeated queries."""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key_data = {"args": args, "kwargs": kwargs}
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if asyncio.get_event_loop().time() - timestamp < ttl_seconds:
                    return result
            
            # Execute and cache
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, asyncio.get_event_loop().time())
            
            return result
        
        return wrapper
    return decorator

# Usage
@cache_results(ttl_seconds=1800)  # 30 minutes
async def cached_agent_run(agent, task):
    return await agent.run(task)
```

### Parallel Processing

```python
import asyncio
from saplings import Agent

async def parallel_processing_example():
    """Process multiple tasks in parallel."""
    config = AgentConfig.standard("openai", "gpt-4o")
    
    # Create multiple agent instances for parallel processing
    agents = [Agent(config) for _ in range(3)]
    
    tasks = [
        "Summarize recent AI developments",
        "Explain quantum computing basics",
        "Write a Python sorting algorithm"
    ]
    
    # Process tasks in parallel
    results = await asyncio.gather(*[
        agent.run(task) 
        for agent, task in zip(agents, tasks)
    ])
    
    return results
```

## Testing Strategies

### Unit Testing

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from saplings import Agent, AgentConfig

class TestAgent:
    @pytest.fixture
    def mock_config(self):
        return AgentConfig.minimal("mock", "test-model")
    
    @pytest.fixture
    def agent(self, mock_config):
        return Agent(mock_config)
    
    @pytest.mark.asyncio
    async def test_basic_run(self, agent):
        """Test basic agent execution."""
        with patch.object(agent, '_facade') as mock_facade:
            mock_facade.run.return_value = "Test response"
            
            result = await agent.run("Test task")
            
            assert result == "Test response"
            mock_facade.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_tool_integration(self, agent):
        """Test tool registration and usage."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        
        agent.register_tool(mock_tool)
        
        # Verify tool is registered
        assert mock_tool in agent._facade.tool_service.tools
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = AgentConfig("invalid_provider", "test-model")
        
        with pytest.raises(ValueError):
            config.validate()
```

### Integration Testing

```python
import pytest
import tempfile
import shutil
from saplings import Agent, AgentConfig

class TestAgentIntegration:
    @pytest.fixture
    def temp_memory_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self, temp_memory_path):
        """Test that memory persists across agent instances."""
        config = AgentConfig.minimal(
            "mock", "test-model",
            memory_path=temp_memory_path
        )
        
        # Create first agent and add document
        agent1 = Agent(config)
        agent1.add_document("Test document", {"source": "test"})
        
        # Create second agent with same memory path
        agent2 = Agent(config)
        docs = await agent2.retrieve("Test")
        
        assert len(docs) > 0
        assert "Test document" in docs[0]["content"]
    
    @pytest.mark.asyncio
    async def test_tool_execution_flow(self):
        """Test complete tool execution flow."""
        config = AgentConfig.minimal("mock", "test-model")
        agent = Agent(config)
        
        @tool(name="test_calculator", description="Simple calculator")
        def calculator(a: int, b: int, operation: str = "add") -> int:
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        agent.register_tool(calculator)
        
        # Mock the model response to use the tool
        with patch.object(agent._facade.execution_service, 'execute') as mock_execute:
            mock_response = Mock()
            mock_response.tool_calls = [{
                "function": {
                    "name": "test_calculator",
                    "arguments": '{"a": 2, "b": 3, "operation": "add"}'
                }
            }]
            mock_execute.return_value = mock_response
            
            result = await agent.run("Calculate 2 + 3")
            
            # Verify tool was called
            mock_execute.assert_called_once()
```

### Performance Testing

```python
import time
import asyncio
import statistics
from saplings import Agent, AgentConfig

async def performance_test():
    """Benchmark agent performance."""
    config = AgentConfig.minimal("mock", "test-model")
    agent = Agent(config)
    
    tasks = ["Simple task"] * 10
    times = []
    
    for task in tasks:
        start_time = time.time()
        await agent.run(task)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    
    print(f"Average execution time: {avg_time:.2f}s")
    print(f"Median execution time: {median_time:.2f}s")
    print(f"Min execution time: {min(times):.2f}s")
    print(f"Max execution time: {max(times):.2f}s")
    
    return {
        "average": avg_time,
        "median": median_time,
        "min": min(times),
        "max": max(times)
    }
```

## Troubleshooting

### Common Issues

#### 1. Container Not Configured

**Problem:** `Container is not configured` error

**Solution:**
```python
from saplings.api.di import configure_container, container

# Ensure container is configured before creating agents
config = AgentConfig.for_openai("gpt-4o")
configure_container(config)

# Or let Agent auto-configure
agent = Agent(provider="openai", model_name="gpt-4o")
```

#### 2. Memory Path Issues

**Problem:** `Cannot create memory directory` error

**Solution:**
```python
import os
from pathlib import Path

# Ensure directory exists and is writable
memory_path = "./agent_memory"
Path(memory_path).mkdir(parents=True, exist_ok=True)

# Or use absolute path
memory_path = os.path.abspath("./agent_memory")
config = AgentConfig("openai", "gpt-4o", memory_path=memory_path)
```

#### 3. Tool Import Errors

**Problem:** Tool dependencies not available

**Solution:**
```python
from saplings.api.tools import is_browser_tools_available

# Check availability before using
if is_browser_tools_available():
    agent = Agent(tools=["GoToTool", "ClickTool"])
else:
    print("Browser tools not available, install with: pip install saplings[browser]")
    agent = Agent()  # Create without browser tools
```

#### 4. API Key Issues

**Problem:** Authentication errors with cloud providers

**Solution:**
```python
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-key"
os.environ["ANTHROPIC_API_KEY"] = "your-key"

# Or pass explicitly
config = AgentConfig.for_openai("gpt-4o", api_key="your-key")

# Validate configuration
validation = config.check_dependencies()
if not validation.all_available:
    print(f"Missing dependencies: {validation.missing_dependencies}")
```

### Debugging Tools

#### Enable Detailed Logging

```python
import logging
from saplings.api.utils import setup_logging

# Enable debug logging
setup_logging(level="DEBUG", format="structured")

# Or use standard logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("saplings")
```

#### Monitor Execution

```python
from saplings import Agent

agent = Agent(
    provider="openai",
    model_name="gpt-4o",
    enable_monitoring=True
)

# Run task
result = await agent.run("Test task")

# Get execution details
traces = agent.get_execution_traces()
metrics = agent.get_performance_metrics()

print(f"Execution traces: {len(traces)}")
print(f"Total tokens used: {metrics.get('total_tokens', 0)}")
print(f"Total cost: ${metrics.get('total_cost', 0):.4f}")
```

#### Validate Configuration

```python
config = AgentConfig.for_openai("gpt-4o")

# Comprehensive validation
validation = config.validate()
if not validation.is_valid:
    print(f"Validation failed: {validation.message}")
    for suggestion in validation.suggestions:
        print(f"  - {suggestion}")

# Check dependencies
deps = config.check_dependencies()
if not deps.all_available:
    print(f"Missing dependencies: {deps.missing_dependencies}")

# Get configuration explanation
print(config.explain())
```

## Advanced Topics

### Custom Service Implementation

```python
from saplings.core.interfaces import IMemoryManager
from saplings.api.di import container

class CustomMemoryService(IMemoryManager):
    """Custom memory implementation with external database."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        # Initialize database connection
    
    def add_document(self, content: str, metadata: dict = None) -> str:
        # Custom implementation
        pass
    
    def get_document(self, doc_id: str) -> dict:
        # Custom implementation
        pass
    
    # ... implement other required methods

# Register custom service
container.register("MemoryService", CustomMemoryService("postgresql://..."))
```

### Plugin Development

```python
from saplings.core.plugin import Plugin

class MyPlugin(Plugin):
    """Custom plugin for extending Saplings functionality."""
    
    def __init__(self):
        super().__init__("my_plugin", "1.0.0")
    
    def initialize(self, container):
        """Initialize plugin with dependency container."""
        # Register custom services, tools, validators, etc.
        self.register_tools(container)
        self.register_validators(container)
    
    def register_tools(self, container):
        """Register plugin-specific tools."""
        from .tools import CustomTool
        container.register_tool(CustomTool())
    
    def register_validators(self, container):
        """Register plugin-specific validators."""
        from .validators import CustomValidator
        container.register_validator(CustomValidator())

# Load plugin
plugin = MyPlugin()
plugin.initialize(container)
```

This developer guide provides the foundation for building robust applications with Saplings. Remember to follow the patterns and best practices outlined here for optimal performance and maintainability.