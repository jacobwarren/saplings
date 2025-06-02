# Saplings Configuration Guide

This guide provides comprehensive documentation for configuring Saplings agents, including predefined configurations, customization patterns, and advanced configuration techniques.

## Table of Contents

- [Overview](#overview)
- [Predefined Configurations](#predefined-configurations)
- [Provider-Specific Configurations](#provider-specific-configurations)
- [Customizing Configurations](#customizing-configurations)
- [Configuration Reference](#configuration-reference)
- [Advanced Configuration Patterns](#advanced-configuration-patterns)
- [Configuration Best Practices](#configuration-best-practices)

## Overview

Saplings provides multiple ways to configure agents:

1. **Predefined Presets**: Ready-to-use configurations for common scenarios
2. **Provider-Specific Presets**: Optimized configurations for specific LLM providers
3. **Builder Customization**: Fine-tune any aspect using the fluent builder API
4. **Configuration Objects**: Advanced customization using AgentConfig directly

## Predefined Configurations

### Configuration Preset Comparison

| Feature | Minimal | Standard | Full-Featured |
|---------|---------|----------|---------------|
| **GASA** | ❌ Disabled | ✅ Enabled | ✅ Enabled (Advanced) |
| **Monitoring** | ❌ Disabled | ✅ Enabled | ✅ Enabled |
| **Self-Healing** | ❌ Disabled | ❌ Disabled | ✅ Enabled |
| **Tool Factory** | ❌ Disabled | ✅ Enabled | ✅ Enabled |
| **Planning Strategy** | Fixed | Proportional | Dynamic |
| **Budget Management** | Basic | Standard | Advanced |
| **Validation** | Basic | Execution | Judge-based |
| **Memory Features** | Basic | Standard | Enhanced |
| **Use Case** | Simple tasks | Most applications | Complex workflows |

### Minimal Configuration

**When to use:** Simple queries, prototyping, learning, resource-constrained environments.

```python
from saplings import AgentBuilder

# Minimal preset
agent = AgentBuilder.minimal("openai", "gpt-4o").build()

# What this includes:
# - Basic model connection
# - Simple memory storage
# - No advanced features
# - Minimal resource usage
```

**Features included:**
- ❌ GASA disabled for simplicity
- ❌ Monitoring disabled
- ❌ Self-healing disabled  
- ❌ Tool factory disabled
- ✅ Basic execution validation
- ✅ Fixed planning strategy
- ✅ Essential tools only

**Resource usage:** Low CPU, minimal memory

### Standard Configuration

**When to use:** Most production applications, balanced feature set, general-purpose agents.

```python
from saplings import AgentBuilder

# Standard preset
agent = AgentBuilder.standard("openai", "gpt-4o").build()

# What this includes:
# - GASA for better performance
# - Monitoring for observability  
# - Tool factory for dynamic capabilities
# - Balanced resource usage
```

**Features included:**
- ✅ GASA enabled (binary strategy)
- ✅ Monitoring enabled
- ❌ Self-healing disabled (stability over automation)
- ✅ Tool factory enabled
- ✅ Execution validation
- ✅ Proportional planning strategy
- ✅ Standard tool set

**Resource usage:** Moderate CPU, standard memory

### Full-Featured Configuration

**When to use:** Complex workflows, maximum capabilities, research applications, when performance is critical.

```python
from saplings import AgentBuilder

# Full-featured preset
agent = AgentBuilder.full_featured("openai", "gpt-4o").build()

# What this includes:
# - All advanced features enabled
# - Maximum GASA performance
# - Self-healing capabilities
# - Advanced planning and validation
```

**Features included:**
- ✅ GASA enabled with advanced settings (3 hops, learned strategy)
- ✅ Full monitoring and tracing
- ✅ Self-healing with retry logic
- ✅ Tool factory with sandboxing
- ✅ Judge-based validation
- ✅ Dynamic planning with budget overflow
- ✅ Enhanced memory and retrieval (20 documents)
- ✅ All available tools

**Resource usage:** High CPU, more memory

### Preset Usage Examples

```python
from saplings import AgentBuilder

# Choose based on your needs:

# For learning and simple tasks
learning_agent = AgentBuilder.minimal("openai", "gpt-4o") \
    .with_api_key("your-key") \
    .build()

# For most applications  
production_agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_api_key("your-key") \
    .with_memory_path("./prod_memory") \
    .build()

# For complex research workflows
research_agent = AgentBuilder.full_featured("openai", "gpt-4o") \
    .with_api_key("your-key") \
    .with_memory_path("./research_memory") \
    .with_planner_total_budget(10.0) \
    .build()
```

## Provider-Specific Configurations

### OpenAI Configuration

**Optimizations for OpenAI models:**

```python
agent = AgentBuilder.for_openai("gpt-4o").build()

# What this optimizes:
# - GASA with shadow model for tokenization
# - Prompt composer for better context handling
# - Appropriate token limits (4096)
# - Temperature and sampling optimized for GPT models
```

**OpenAI-specific features:**
- ✅ GASA shadow model: `Qwen/Qwen3-0.6B` for tokenization
- ✅ Prompt composer fallback
- ✅ Optimized for OpenAI's attention patterns
- ✅ Token limits: 4096 max tokens
- ✅ Best practices for GPT model family

### Anthropic Configuration

**Optimizations for Claude models:**

```python
agent = AgentBuilder.for_anthropic("claude-3-opus").build()

# What this optimizes:
# - GASA tuned for Claude's architecture
# - Prompt formatting for Claude's preferences
# - Token limits appropriate for Claude (4096)
# - Constitutional AI alignment considerations
```

**Anthropic-specific features:**
- ✅ GASA shadow model for tokenization alignment
- ✅ Prompt composer optimized for Claude
- ✅ Constitutional AI-friendly validation
- ✅ Claude-optimized conversation patterns
- ✅ Token limits: 4096 max tokens

### vLLM Configuration

**Optimizations for self-hosted models:**

```python
agent = AgentBuilder.for_vllm("Qwen/Qwen3-7B-Instruct").build()

# What this optimizes:
# - No shadow model (uses same model for tokenization)
# - Block diagonal fallback (more efficient for local models)
# - GPU memory management settings
# - Tensor parallelism configuration
```

**vLLM-specific features:**
- ❌ No shadow model (efficiency)
- ✅ Block diagonal GASA fallback
- ✅ GPU memory utilization: 80%
- ✅ Tensor parallelism: 1 GPU by default
- ✅ Trust remote code: enabled
- ✅ Local model optimizations

### Provider Comparison

| Provider | Shadow Model | GASA Fallback | Token Limit | Key Optimization |
|----------|--------------|---------------|-------------|------------------|
| **OpenAI** | ✅ Qwen/Qwen3-0.6B | Prompt Composer | 4096 | API efficiency |
| **Anthropic** | ✅ Qwen/Qwen3-0.6B | Prompt Composer | 4096 | Constitutional AI |
| **vLLM** | ❌ Same model | Block Diagonal | Model-dependent | Local efficiency |

## Customizing Configurations

### Overriding Preset Settings

You can start with a preset and customize specific aspects:

```python
from saplings import AgentBuilder

# Start with standard preset and customize
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_api_key("your-key") \
    .with_gasa_max_hops(5) \
    .with_planner_total_budget(20.0) \
    .with_self_healing_enabled(True) \
    .with_retrieval_max_documents(30) \
    .build()

# Start with provider preset and add features
agent = AgentBuilder.for_openai("gpt-4o") \
    .with_monitoring_enabled(True) \
    .with_tools(["PythonInterpreterTool", "DuckDuckGoSearchTool"]) \
    .with_memory_path("./custom_memory") \
    .build()
```

### Mixing Preset Philosophies

```python
# Take minimal base but add specific advanced features
agent = AgentBuilder.minimal("openai", "gpt-4o") \
    .with_gasa_enabled(True) \
    .with_monitoring_enabled(True) \
    .build()  # Minimal + performance features

# Take full-featured but reduce resource usage
agent = AgentBuilder.full_featured("openai", "gpt-4o") \
    .with_gasa_max_hops(1) \
    .with_retrieval_max_documents(5) \
    .with_self_healing_enabled(False) \
    .build()  # Full-featured but lighter
```

### Builder Method Categories

#### Core Configuration
```python
agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_api_key("your-key") \
    .build()
```

#### Memory & Storage
```python
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_memory_path("./agent_memory") \
    .with_output_dir("./outputs") \
    .build()
```

#### GASA Configuration
```python
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_gasa_enabled(True) \
    .with_gasa_max_hops(3) \
    .with_gasa_strategy("binary") \
    .with_gasa_fallback("prompt_composer") \
    .with_gasa_shadow_model_enabled(True) \
    .with_gasa_shadow_model_name("Qwen/Qwen3-0.6B") \
    .build()
```

#### Planning & Execution
```python
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_planner_budget_strategy("dynamic") \
    .with_planner_total_budget(5.0) \
    .with_planner_allow_budget_overflow(True) \
    .with_planner_budget_overflow_margin(0.2) \
    .with_executor_validation_type("judge") \
    .build()
```

#### Tools & Capabilities
```python
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_tools(["PythonInterpreterTool", "DuckDuckGoSearchTool"]) \
    .with_tool_factory_enabled(True) \
    .with_tool_factory_sandbox_enabled(True) \
    .with_allowed_imports(["pandas", "numpy", "matplotlib"]) \
    .build()
```

#### Monitoring & Validation
```python
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_monitoring_enabled(True) \
    .with_self_healing_enabled(True) \
    .with_self_healing_max_retries(5) \
    .build()
```

#### Retrieval Configuration
```python
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_retrieval_max_documents(15) \
    .with_retrieval_entropy_threshold(0.15) \
    .build()
```

#### Model Parameters
```python
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_temperature(0.8) \
    .with_max_tokens(3000) \
    .with_model_parameters({
        "top_p": 0.95,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }) \
    .build()
```

#### Multi-Modal Support
```python
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_supported_modalities(["text", "image", "audio"]) \
    .build()
```

### Advanced Configuration with AgentConfig

For complex scenarios, you can use AgentConfig directly:

```python
from saplings import AgentConfig, AgentBuilder

# Create custom config object
config = AgentConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-key",
    
    # Advanced GASA settings
    enable_gasa=True,
    gasa_max_hops=4,
    gasa_strategy="learned",
    gasa_fallback="hybrid",
    gasa_shadow_model=True,
    gasa_shadow_model_name="microsoft/DialoGPT-medium",
    
    # Custom planning
    planner_budget_strategy="ml_predicted",
    planner_total_budget=15.0,
    planner_allow_budget_overflow=True,
    planner_budget_overflow_margin=0.3,
    
    # Advanced validation
    executor_validation_type="multi_judge",
    
    # Custom imports for tool factory
    allowed_imports=[
        "torch", "transformers", "sklearn", 
        "pandas", "numpy", "matplotlib", "seaborn"
    ],
    
    # Multi-modal configuration
    supported_modalities=["text", "image", "audio", "video"],
    
    # Advanced model parameters
    **{
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_tokens": 4096,
        "custom_parameter": "value"
    }
)

# Use with builder
agent = AgentBuilder().with_config(config).build()

# Or create agent directly
agent = Agent(config=config)
```

## Configuration Reference

### Complete Builder Method Reference

#### Core Methods
- `.with_provider(provider: str)` - Set model provider
- `.with_model_name(model_name: str)` - Set model name  
- `.with_api_key(api_key: str)` - Set API key
- `.with_config(config: AgentConfig)` - Use custom config object

#### Memory & Storage Methods
- `.with_memory_path(path: str)` - Set memory storage path
- `.with_output_dir(path: str)` - Set output directory

#### GASA Methods
- `.with_gasa_enabled(enabled: bool)` - Enable/disable GASA
- `.with_gasa_max_hops(hops: int)` - Set maximum attention hops
- `.with_gasa_strategy(strategy: str)` - Set GASA strategy ("binary", "soft", "learned")
- `.with_gasa_fallback(fallback: str)` - Set fallback strategy
- `.with_gasa_shadow_model_enabled(enabled: bool)` - Enable shadow model
- `.with_gasa_shadow_model_name(name: str)` - Set shadow model name
- `.with_gasa_prompt_composer_enabled(enabled: bool)` - Enable prompt composer

#### Planning Methods
- `.with_planner_budget_strategy(strategy: str)` - Set budget strategy
- `.with_planner_total_budget(budget: float)` - Set total budget in USD
- `.with_planner_allow_budget_overflow(allow: bool)` - Allow budget overflow
- `.with_planner_budget_overflow_margin(margin: float)` - Set overflow margin

#### Execution Methods
- `.with_executor_validation_type(type: str)` - Set validation type
- `.with_temperature(temp: float)` - Set model temperature
- `.with_max_tokens(tokens: int)` - Set max tokens

#### Tool Methods
- `.with_tools(tools: List[str])` - Set tool list
- `.with_custom_tools(tools: List[callable])` - Add custom tools
- `.with_tool_factory_enabled(enabled: bool)` - Enable tool factory
- `.with_tool_factory_sandbox_enabled(enabled: bool)` - Enable sandboxing
- `.with_allowed_imports(imports: List[str])` - Set allowed imports

#### Monitoring Methods
- `.with_monitoring_enabled(enabled: bool)` - Enable monitoring
- `.with_self_healing_enabled(enabled: bool)` - Enable self-healing
- `.with_self_healing_max_retries(retries: int)` - Set retry limit

#### Retrieval Methods
- `.with_retrieval_max_documents(max_docs: int)` - Set max documents
- `.with_retrieval_entropy_threshold(threshold: float)` - Set entropy threshold

#### Multi-Modal Methods
- `.with_supported_modalities(modalities: List[str])` - Set supported modalities

#### Advanced Methods
- `.with_model_parameters(params: dict)` - Set additional model parameters
- `.with_validators(validators: List)` - Set custom validators

### Configuration Values Reference

#### GASA Strategies
- `"binary"` - Simple binary attention masks
- `"soft"` - Soft attention weights  
- `"learned"` - Learned attention patterns

#### GASA Fallbacks
- `"block_diagonal"` - Block diagonal attention pattern
- `"prompt_composer"` - Graph-aware prompt composition

#### Planning Strategies
- `"fixed"` - Fixed budget per step
- `"proportional"` - Budget proportional to complexity
- `"dynamic"` - Dynamic budget allocation

#### Validation Types
- `"basic"` - Basic format validation
- `"execution"` - Execute and validate results
- `"judge"` - LLM-based quality judgment

#### Supported Modalities
- `"text"` - Text input/output
- `"image"` - Image processing
- `"audio"` - Audio processing  
- `"video"` - Video processing

## Advanced Configuration Patterns

### Environment-Specific Configuration

```python
import os
from saplings import AgentBuilder

def get_environment_config():
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "development":
        return AgentBuilder.minimal("openai", "gpt-4o") \
            .with_monitoring_enabled(True)
    
    elif env == "staging":
        return AgentBuilder.standard("openai", "gpt-4o") \
            .with_monitoring_enabled(True) \
            .with_self_healing_enabled(True)
    
    elif env == "production":
        return AgentBuilder.full_featured("openai", "gpt-4o") \
            .with_planner_total_budget(50.0) \
            .with_memory_path("/var/lib/saplings")
    
    else:
        raise ValueError(f"Unknown environment: {env}")

agent = get_environment_config().build()
```

### Task-Specific Configuration

```python
class AgentConfigurations:
    @staticmethod
    def for_research():
        return AgentBuilder.full_featured("openai", "gpt-4o") \
            .with_tools([
                "DuckDuckGoSearchTool",
                "WikipediaSearchTool", 
                "PythonInterpreterTool"
            ]) \
            .with_retrieval_max_documents(50) \
            .with_planner_total_budget(20.0)
    
    @staticmethod
    def for_coding():
        return AgentBuilder.standard("anthropic", "claude-3-opus") \
            .with_tools(["PythonInterpreterTool"]) \
            .with_tool_factory_enabled(True) \
            .with_executor_validation_type("execution") \
            .with_allowed_imports([
                "os", "sys", "json", "re", "math",
                "pandas", "numpy", "requests"
            ])
    
    @staticmethod  
    def for_content_creation():
        return AgentBuilder.standard("openai", "gpt-4o") \
            .with_supported_modalities(["text", "image"]) \
            .with_tools([
                "DuckDuckGoSearchTool",
                "PythonInterpreterTool"
            ]) \
            .with_temperature(0.8)

# Usage
research_agent = AgentConfigurations.for_research().build()
coding_agent = AgentConfigurations.for_coding().build()
content_agent = AgentConfigurations.for_content_creation().build()
```

### Configuration Inheritance

```python
from saplings import AgentBuilder

# Base configuration
base_config = AgentBuilder.standard("openai", "gpt-4o") \
    .with_monitoring_enabled(True) \
    .with_memory_path("./shared_memory")

# Specialized configurations inheriting from base
research_agent = base_config \
    .with_tools(["DuckDuckGoSearchTool", "WikipediaSearchTool"]) \
    .with_retrieval_max_documents(30) \
    .build()

coding_agent = base_config \
    .with_tools(["PythonInterpreterTool"]) \
    .with_tool_factory_enabled(True) \
    .build()
```

## Configuration Best Practices

### 1. Start with Presets

```python
# Good: Start with appropriate preset
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_custom_modifications() \
    .build()

# Avoid: Building from scratch unless necessary
agent = AgentBuilder() \
    .with_provider("openai") \
    .with_model_name("gpt-4o") \
    .with_gasa_enabled(True) \
    .with_monitoring_enabled(True) \
    # ... many more settings
```

### 2. Use Provider-Specific Presets

```python
# Good: Leverage provider optimizations
agent = AgentBuilder.for_openai("gpt-4o") \
    .with_tools(["PythonInterpreterTool"]) \
    .build()

# Less optimal: Generic configuration
agent = AgentBuilder.standard("openai", "gpt-4o") \
    .with_gasa_shadow_model_enabled(True) \
    .with_gasa_fallback("prompt_composer") \
    # ... manually setting OpenAI optimizations
```

### 3. Environment-Specific Settings

```python
# Good: Different configs per environment
def create_agent():
    if os.getenv("DEBUG"):
        return AgentBuilder.minimal("openai", "gpt-4o") \
            .with_monitoring_enabled(True)
    else:
        return AgentBuilder.standard("openai", "gpt-4o")
```

### 4. Document Custom Configurations

```python
def create_specialized_agent():
    """
    Creates an agent optimized for financial analysis tasks.
    
    Configuration choices:
    - Claude model for better reasoning
    - Extended budget for complex calculations  
    - Specialized tools for financial data
    - Enhanced validation for accuracy
    """
    return AgentBuilder.for_anthropic("claude-3-opus") \
        .with_tools(["PythonInterpreterTool", "FinancialDataTool"]) \
        .with_planner_total_budget(15.0) \
        .with_executor_validation_type("judge") \
        .build()
```

### 5. Validate Configurations

```python
def create_production_agent():
    agent_builder = AgentBuilder.full_featured("openai", "gpt-4o")
    
    # Validate configuration before building
    config = agent_builder._build_config()  # Internal method
    validation = config.validate()
    
    if not validation.is_valid:
        raise ValueError(f"Invalid configuration: {validation.message}")
    
    return agent_builder.build()
```

This configuration guide provides users with a clear understanding of all available options and how to use them effectively!