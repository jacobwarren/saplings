# Saplings Examples

This directory contains comprehensive examples demonstrating how to use the Saplings framework for building intelligent agents with various capabilities.

## Prerequisites

1. **Python 3.8+** with asyncio support
2. **Saplings installed**: `pip install saplings`
3. **API Keys** (for relevant examples):
   - OpenAI: Set `OPENAI_API_KEY` environment variable
   - Anthropic: Set `ANTHROPIC_API_KEY` environment variable

## Example Categories

### 🚀 Getting Started
- **[01_basic_agent_usage.py](01_basic_agent_usage.py)** - Simple agent creation and usage patterns
- **[02_advanced_agent_usage.py](02_advanced_agent_usage.py)** - Advanced configurations and features

### ⚡ Performance Optimization
- **[03_gasa_openai_example.py](03_gasa_openai_example.py)** - GASA optimization for OpenAI models
- **[04_gasa_qwen3_transformers_example.py](04_gasa_qwen3_transformers_example.py)** - GASA with local Qwen3 models

### 🛠️ Specialized Applications
- **[05_code_repo_assistant.py](05_code_repo_assistant.py)** - Code analysis and repository management
- **[06_dynamic_tool_creation.py](06_dynamic_tool_creation.py)** - Runtime tool generation and registration
- **[07_faiss_vector_store_example.py](07_faiss_vector_store_example.py)** - Advanced vector storage with FAISS
- **[08_multimodality_example.py](08_multimodality_example.py)** - Multi-modal processing (text, image, audio)
- **[09_plugin_registration.py](09_plugin_registration.py)** - Plugin system and extensibility

### 🔬 Real-World Applications
- **[10_huggingface_research_analyzer.py](10_huggingface_research_analyzer.py)** - End-to-end research paper analysis
- **[11_self_healing_monitor.py](11_self_healing_monitor.py)** - Self-healing and resilience patterns
- **[12_self_improving_agent.py](12_self_improving_agent.py)** - Self-improvement and adaptation
- **[13_existing_tool_usage.py](13_existing_tool_usage.py)** - Using pre-built tools effectively

## Quick Start

1. **Basic Usage**:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   python 01_basic_agent_usage.py
   ```

2. **Advanced Features**:
   ```bash
   python 02_advanced_agent_usage.py
   ```

3. **GASA Optimization**:
   ```bash
   python 03_gasa_openai_example.py
   ```

## Example Features Matrix

| Example | GASA | Tools | Memory | Monitoring | Self-Healing | Multi-Modal |
|---------|------|-------|--------|------------|--------------|-------------|
| 01_basic | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 02_advanced | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| 03_gasa_openai | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| 04_gasa_qwen3 | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| 05_code_repo | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| 06_dynamic_tools | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 07_faiss | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| 08_multimodal | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ |
| 09_plugins | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| 10_research | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| 11_self_healing | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| 12_self_improving | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| 13_existing_tools | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |

## Key Concepts Demonstrated

### Graph-Aligned Sparse Attention (GASA)
GASA optimizes performance by focusing attention on relevant context based on document relationships:
- **Binary Strategy**: Fast attention masking for quick responses
- **Mask Strategy**: Detailed attention control for complex reasoning
- **Shadow Models**: Lightweight models guide attention for larger models
- **Fallback Strategies**: Graceful degradation when GASA limits are exceeded

### Tool Integration
Examples show different approaches to tool usage:
- **Pre-built Tools**: Using existing Saplings tools
- **Custom Tools**: Creating domain-specific functionality
- **Dynamic Tools**: Runtime tool generation and loading
- **Tool Composition**: Combining multiple tools for complex workflows

### Memory Management
Demonstrates various memory and retrieval patterns:
- **Basic Memory**: Simple document storage and retrieval
- **Vector Stores**: Advanced similarity search with FAISS
- **Dependency Graphs**: Understanding document relationships
- **Memory Optimization**: Efficient storage for large knowledge bases

### Self-Healing and Resilience
Shows how agents can recover from failures:
- **Automatic Retry**: Configurable retry strategies
- **Error Detection**: Identifying and categorizing failures
- **Adaptive Behavior**: Learning from failures to prevent recurrence
- **Graceful Degradation**: Maintaining functionality under stress

## Environment Setup

### For OpenAI Examples
```bash
export OPENAI_API_KEY="sk-..."
```

### For Anthropic Examples
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### For Local Model Examples
```bash
# Install transformers for local models
pip install transformers torch

# Or set up vLLM server
pip install vllm
vllm serve Qwen/Qwen3-7B-Instruct
```

### For FAISS Examples
```bash
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
```

## Output and Artifacts

Examples create various outputs:
- **Memory Stores**: `./agent_memory/`, `./gasa_memory/`, etc.
- **Monitoring Data**: `./monitoring_output/`, visualization files
- **Generated Code**: Dynamic tools and plugins
- **Analysis Results**: Code reviews, documentation, refactoring suggestions

## Common Patterns

### Agent Builder Usage
```python
from saplings import AgentBuilder

# Provider-specific optimizations
agent = AgentBuilder.for_openai("gpt-4o", api_key="...").build()
agent = AgentBuilder.for_anthropic("claude-3-opus", api_key="...").build()
agent = AgentBuilder.for_vllm("Qwen/Qwen3-7B-Instruct").build()

# Configuration presets
agent = AgentBuilder.minimal("openai", "gpt-4o").build()
agent = AgentBuilder.standard("openai", "gpt-4o").build()
agent = AgentBuilder.full_featured("openai", "gpt-4o").build()

# Custom configuration
agent = (AgentBuilder()
    .with_provider("openai")
    .with_model_name("gpt-4o")
    .with_gasa_enabled(True)
    .with_memory_path("./memory")
    .with_tools([...])
    .build())
```

### Error Handling
```python
from saplings.core.exceptions import ModelError, MemoryError

try:
    response = await agent.run("Your query here")
except ModelError as e:
    print(f"Model error: {e}")
except MemoryError as e:
    print(f"Memory error: {e}")
```

### Custom Tools
```python
from saplings.api.tools import tool

@tool(name="my_tool", description="Does something useful")
def my_function(param: str) -> str:
    """Tool function with proper type hints."""
    return f"Processed: {param}"

agent = AgentBuilder.for_openai("gpt-4o").with_tools([my_function]).build()
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**:
   ```
   Error: OpenAI API key not found
   Solution: export OPENAI_API_KEY="your-key"
   ```

2. **Local Model Not Available**:
   ```
   Error: Model server not reachable
   Solution: Start vLLM server or install transformers
   ```

3. **Memory Path Issues**:
   ```
   Error: Permission denied creating memory directory
   Solution: Ensure write permissions or use different path
   ```

4. **GASA Configuration Errors**:
   ```
   Error: Invalid GASA strategy
   Solution: Use "binary", "mask", or check documentation
   ```

### Performance Tips

1. **GASA Optimization**:
   - Use `binary` strategy for speed, `mask` for quality
   - Adjust `max_hops` based on document complexity
   - Enable shadow models for API cost reduction

2. **Memory Management**:
   - Use appropriate chunk sizes for your content
   - Enable FAISS for large document collections
   - Monitor memory usage with large knowledge bases

3. **Tool Performance**:
   - Cache expensive tool operations
   - Use async tools for I/O operations
   - Limit tool execution time with timeouts

## Next Steps

1. **Start with Basic Examples**: Run 01 and 02 to understand fundamentals
2. **Explore GASA**: Try examples 03 and 04 for performance optimization
3. **Build Applications**: Use specialized examples as templates
4. **Create Custom Tools**: Extend functionality for your use case
5. **Deploy in Production**: Review monitoring and self-healing examples

## Contributing

To add new examples:
1. Follow the naming convention: `##_descriptive_name.py`
2. Include comprehensive docstrings and error handling
3. Add entry to this README with feature matrix
4. Test with multiple configurations
5. Document any special requirements

For questions or issues, please refer to the main Saplings documentation or create an issue in the repository.