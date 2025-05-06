# GASA with Third-Party LLMs

This document explains how to use Graph-Aligned Sparse Attention (GASA) with third-party LLM APIs like OpenAI and Anthropic, which don't expose low-level attention mechanisms.

## Overview

GASA (Graph-Aligned Sparse Attention) is a technique that improves efficiency and grounding in language models by focusing attention on relevant context based on document relationships. While GASA works best with models that allow direct manipulation of attention masks (like local models served through vLLM), Saplings provides several alternative approaches for using GASA with third-party LLM APIs:

1. **Shadow Model Tokenization**: Uses a small local model for tokenization and mask generation
2. **Graph-Aware Prompt Composition**: Structures prompts based on graph relationships
3. **Block-Diagonal Packing**: Reorders chunks to create a block-diagonal structure

## Shadow Model Tokenization

The shadow model approach uses a small local model for tokenization and mask generation:

```python
from saplings.tokenizers import ShadowModelTokenizer
from saplings.gasa import GASAConfig
from saplings import Agent, AgentConfig

# Create an agent with GASA and shadow model
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_gasa=True,
        gasa_max_hops=2,
        gasa_strategy="binary",
        gasa_shadow_model=True,
        gasa_shadow_model_name="Qwen/Qwen3-0.6B",
    )
)
```

### How Shadow Model Tokenization Works

1. **Shadow Model Loading**: A small local model (default: Qwen/Qwen3-0.6B) is loaded for tokenization
2. **Token Mapping**: The shadow model tokenizes the prompt and maps tokens to document chunks
3. **Mask Generation**: GASA masks are generated based on the token mapping and document relationships
4. **Prompt Restructuring**: The prompt is restructured based on the GASA masks to guide the third-party LLM's attention

### Configuration Options

```python
class GASAConfig:
    # Shadow model configuration
    enable_shadow_model: bool = False  # Whether to use shadow model for tokenization
    shadow_model_name: str = "Qwen/Qwen3-0.6B"  # Shadow model name
    shadow_model_device: str = "cpu"  # Shadow model device
    fallback_to_simple: bool = True  # Whether to fall back to simple tokenizer if shadow model fails
```

## Graph-Aware Prompt Composition

The prompt composer approach structures prompts based on graph relationships:

```python
from saplings.gasa import GASAConfig
from saplings import Agent, AgentConfig

# Create an agent with GASA and prompt composer
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_gasa=True,
        gasa_max_hops=2,
        gasa_strategy="binary",
        gasa_fallback="prompt_composer",
        gasa_prompt_composer=True,
    )
)
```

### How Prompt Composition Works

1. **Graph Analysis**: The document dependency graph is analyzed to identify relationships
2. **Chunk Ordering**: Chunks are ordered based on their relationships in the graph
3. **Focus Tags**: Special tags are added to highlight important chunks
4. **Structured Prompt**: The prompt is structured to guide the model's attention to related chunks

### Configuration Options

```python
class GASAConfig:
    # Prompt composer configuration
    enable_prompt_composer: bool = False  # Whether to use prompt composer
    focus_tags: bool = True  # Whether to use focus tags
    focus_tag_start: str = "<<FOCUS>>"  # Focus tag start
    focus_tag_end: str = "<</FOCUS>>"  # Focus tag end
```

## Block-Diagonal Packing

The block-diagonal packing approach reorders chunks to create a block-diagonal structure:

```python
from saplings.gasa import GASAConfig
from saplings import Agent, AgentConfig

# Create an agent with GASA and block-diagonal packing
agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_gasa=True,
        gasa_max_hops=2,
        gasa_strategy="binary",
        gasa_fallback="block_diagonal",
    )
)
```

### How Block-Diagonal Packing Works

1. **Graph Analysis**: The document dependency graph is analyzed to identify relationships
2. **Chunk Clustering**: Chunks are clustered based on their relationships
3. **Reordering**: Chunks are reordered to create a block-diagonal structure
4. **Prompt Construction**: The prompt is constructed with the reordered chunks

### Configuration Options

```python
class GASAConfig:
    # Block-diagonal packing configuration
    fallback_strategy: str = "block_diagonal"  # Fallback strategy for third-party LLMs
    block_size: int = 512  # Block size for block-diagonal packing
    add_summary_token: bool = True  # Whether to add a summary token
```

## Automatic Selection

Saplings automatically selects the appropriate GASA strategy based on the model provider:

```python
from saplings import Agent, AgentConfig

# For OpenAI
openai_agent = Agent(
    config=AgentConfig(
        provider="openai",
        model_name="gpt-4o",
        enable_gasa=True,
        # Other GASA options...
    )
)

# For Anthropic
anthropic_agent = Agent(
    config=AgentConfig(
        provider="anthropic",
        model_name="claude-3-opus-20240229",
        enable_gasa=True,
        # Other GASA options...
    )
)

# For vLLM (uses native tokenizer and standard GASA)
vllm_agent = Agent(
    config=AgentConfig(
        provider="vllm",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        enable_gasa=True,
        # Other GASA options...
    )
)
```

## Example: GASA with OpenAI

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph
from saplings.gasa import GASAConfig

async def main():
    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Add documents
    await memory.add_document(
        "Graph-Aligned Sparse Attention (GASA) is a technique that improves efficiency and grounding in language models by focusing attention on relevant context based on document relationships."
    )
    await memory.add_document(
        "GASA injects a binary attention mask—derived from the retrieval dependency graph—into each transformer layer, permitting full attention only between tokens whose source chunks are ≤ h hops apart in the graph."
    )
    await memory.add_document(
        "For third-party LLM APIs like OpenAI and Anthropic that don't expose low-level attention mechanisms, GASA provides alternative approaches like shadow model tokenization and graph-aware prompt composition."
    )

    # Build dependency graph
    await graph.build_from_memory(memory)

    # Create an agent with GASA and shadow model
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_gasa=True,
            gasa_max_hops=2,
            gasa_strategy="binary",
            gasa_fallback="prompt_composer",
            gasa_shadow_model=True,
            gasa_shadow_model_name="Qwen/Qwen3-0.6B",
            gasa_prompt_composer=True,
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Run a query
    result = await agent.run("Explain how GASA works with third-party LLMs like OpenAI.")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Example: GASA with Anthropic

```python
import asyncio
from saplings import Agent, AgentConfig
from saplings.memory import MemoryStore, DependencyGraph
from saplings.gasa import GASAConfig

async def main():
    # Create memory components
    memory = MemoryStore()
    graph = DependencyGraph()

    # Add documents
    await memory.add_document(
        "Graph-Aligned Sparse Attention (GASA) is a technique that improves efficiency and grounding in language models by focusing attention on relevant context based on document relationships."
    )
    await memory.add_document(
        "GASA injects a binary attention mask—derived from the retrieval dependency graph—into each transformer layer, permitting full attention only between tokens whose source chunks are ≤ h hops apart in the graph."
    )
    await memory.add_document(
        "For third-party LLM APIs like OpenAI and Anthropic that don't expose low-level attention mechanisms, GASA provides alternative approaches like shadow model tokenization and graph-aware prompt composition."
    )

    # Build dependency graph
    await graph.build_from_memory(memory)

    # Create an agent with GASA and shadow model
    agent = Agent(
        config=AgentConfig(
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            enable_gasa=True,
            gasa_max_hops=2,
            gasa_strategy="binary",
            gasa_fallback="prompt_composer",
            gasa_shadow_model=True,
            gasa_shadow_model_name="Qwen/Qwen3-0.6B",
            gasa_prompt_composer=True,
        )
    )

    # Set memory components
    agent.memory_store = memory
    agent.dependency_graph = graph

    # Run a query
    result = await agent.run("Explain how GASA works with third-party LLMs like Anthropic.")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Considerations

When using GASA with third-party LLMs, consider the following performance implications:

1. **Shadow Model Overhead**: Loading a shadow model adds some overhead, but the model is small (0.6B parameters) and runs on CPU
2. **Prompt Composition Overhead**: Graph-aware prompt composition adds minimal overhead
3. **Block-Diagonal Packing Overhead**: Block-diagonal packing adds some computational overhead for reordering chunks
4. **API Latency**: The dominant factor in performance is usually the API latency, not the GASA overhead

## Best Practices

1. **Choose the Right Approach**: Use prompt composition for most cases, shadow model for complex documents, and block-diagonal packing for large documents
2. **Optimize Graph Structure**: Ensure your dependency graph accurately represents document relationships
3. **Tune GASA Parameters**: Adjust `max_hops` and other parameters based on your specific use case
4. **Cache Results**: Use model caching to avoid redundant API calls
5. **Monitor Performance**: Use the monitoring system to track GASA performance and adjust as needed

## Conclusion

GASA provides significant benefits even when used with third-party LLM APIs that don't expose low-level attention mechanisms. By using shadow model tokenization, graph-aware prompt composition, or block-diagonal packing, you can improve efficiency and grounding in your agents.
