# CLI Interface Guide

**Status: Not Available**

The Saplings framework currently does not provide a Command Line Interface (CLI). All interactions with Saplings must be done through the Python API.

## Available Interfaces

### Python API
The primary way to interact with Saplings is through the Python API:

```python
from saplings import Agent, AgentBuilder, AgentConfig

# Using AgentBuilder (recommended)
agent = AgentBuilder.for_openai("gpt-4o").build()

# Using AgentConfig
config = AgentConfig(provider="openai", model_name="gpt-4o")
agent = Agent(config)
```

### Development Scripts

The project includes several development and benchmarking scripts:

#### GASA Benchmark
```bash
cd benchmarks
python gasa_benchmark.py --model "openai://gpt-4o" --output-dir "./results"
```

#### FAISS Benchmark
```bash
cd benchmarks  
python faiss_benchmark.py --dataset-sizes 1000 10000 --output-dir "./results"
```

## Configuration Management

Configuration is handled programmatically through Python:

```python
from saplings import AgentConfig

# Basic configuration
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-api-key"
)

# Advanced configuration
config = AgentConfig(
    provider="openai", 
    model_name="gpt-4o",
    enable_gasa=True,
    enable_monitoring=True,
    memory_path="./memory",
    output_dir="./output"
)
```

## Future Development

A CLI interface may be added in future versions of Saplings. For now, use the Python API for all agent operations and configuration management.

## See Also

- [Getting Started Guide](GETTING_STARTED.md)
- [API Reference](API_REFERENCE.md) 
- [Configuration Guide](CONFIGURATION_GUIDE.md)