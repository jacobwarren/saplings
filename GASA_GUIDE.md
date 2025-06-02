# GASA (Graph-Aligned Sparse Attention) Guide

GASA (Graph-Aligned Sparse Attention) is a core feature of Saplings that reduces computational cost and improves grounding by focusing the model's attention on relevant context based on document dependency graphs.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Configuration](#configuration)
- [Usage Patterns](#usage-patterns)
- [Fallback Strategies](#fallback-strategies)
- [Advanced Features](#advanced-features)
- [Performance Optimization](#performance-optimization)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

### What is GASA?

GASA injects a **binary attention mask**—derived from the retrieval dependency graph—into each transformer layer, permitting full attention only between tokens whose source chunks are ≤ h hops apart in the graph, while routing others through a lightweight global summary token.

### Key Benefits

- **Computational Efficiency**: Up to 40% fewer FLOPs compared to standard attention
- **Improved Grounding**: Focuses attention on relevant context based on document relationships
- **Flexible Deployment**: Works with local models (vLLM) and third-party APIs (OpenAI, Anthropic)
- **Graph-Aware Processing**: Leverages document dependency structures for better understanding

### Architecture Overview

```
Document Graph → Attention Mask → Model Inference
     ↓               ↓                ↓
[Relationships] → [Sparse Matrix] → [Focused Attention]
```

## Core Concepts

### Graph Distance and Hops

GASA uses graph distance (measured in "hops") to determine which tokens can attend to each other:

- **h=1**: Only directly connected documents can attend to each other
- **h=2**: Documents up to 2 hops apart can attend to each other
- **h=∞**: Full attention (equivalent to no GASA)

### Mask Strategies

```python
from saplings.gasa import MaskStrategy

# Binary masks (0 or 1)
MaskStrategy.BINARY

# Soft masks (continuous values 0-1)
MaskStrategy.SOFT

# Learned masks (requires fine-tuning)
MaskStrategy.LEARNED
```

### Fallback Strategies

For models that don't support sparse attention:

```python
from saplings.gasa import FallbackStrategy

# Block-diagonal packing (reorder tokens)
FallbackStrategy.BLOCK_DIAGONAL

# Graph-aware prompt composition
FallbackStrategy.PROMPT_COMPOSER

# Shadow model for tokenization
FallbackStrategy.SHADOW_MODEL
```

## Configuration

### Basic Configuration

```python
from saplings.gasa import GASAConfig, MaskStrategy, FallbackStrategy

# Default configuration
config = GASAConfig.default()

# Custom configuration
config = GASAConfig(
    enabled=True,
    max_hops=2,
    mask_strategy=MaskStrategy.BINARY,
    fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
    block_size=512,
    overlap=64,
    cache_masks=True,
    visualize=False
)
```

### Provider-Specific Configurations

```python
# Optimized for OpenAI/Anthropic APIs
openai_config = GASAConfig.for_openai()
# Sets: fallback_strategy=PROMPT_COMPOSER, enable_shadow_model=True

# Optimized for Anthropic APIs
anthropic_config = GASAConfig.for_anthropic()

# Optimized for vLLM (local models)
vllm_config = GASAConfig.for_vllm()
# Sets: fallback_strategy=BLOCK_DIAGONAL, enable_shadow_model=False
```

### Configuration Builder Pattern

```python
from saplings.gasa import GASAConfigBuilder

config = (GASAConfigBuilder()
    .with_enabled(True)
    .with_max_hops(3)
    .with_mask_strategy(MaskStrategy.SOFT)
    .with_fallback_strategy(FallbackStrategy.PROMPT_COMPOSER)
    .with_shadow_model(enabled=True, model_name="Qwen/Qwen3-0.6B")
    .with_visualization(enabled=True, output_dir="./gasa_viz")
    .with_caching(enabled=True, cache_dir="./gasa_cache")
    .build())
```

### Advanced Configuration Options

```python
config = GASAConfig(
    # Core settings
    enabled=True,
    max_hops=2,
    mask_strategy=MaskStrategy.BINARY,
    fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
    
    # Global tokens (attend to everything)
    global_tokens=["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
    summary_token="[SUM]",
    add_summary_token=True,
    
    # Block-diagonal packing settings
    block_size=512,
    overlap=64,
    
    # Soft mask settings
    soft_mask_temperature=0.1,
    
    # Caching
    cache_masks=True,
    cache_dir="./gasa_cache",
    
    # Visualization
    visualize=True,
    visualization_dir="./gasa_visualizations",
    
    # Shadow model for third-party APIs
    enable_shadow_model=True,
    shadow_model_name="Qwen/Qwen3-1.8B",
    shadow_model_device="cpu",
    shadow_model_cache_dir="./shadow_cache",
    
    # Prompt composition for third-party APIs
    enable_prompt_composer=True,
    focus_tags=True,
    core_tag="[CORE_CTX]",
    near_tag="[NEAR_CTX]",
    summary_tag="[SUMMARY_CTX]"
)
```

## Usage Patterns

### With AgentBuilder (Recommended)

```python
from saplings import AgentBuilder
from saplings.gasa import GASAConfig, MaskStrategy, FallbackStrategy

# For local models (vLLM)
gasa_config = GASAConfig(
    enabled=True,
    max_hops=2,
    mask_strategy=MaskStrategy.BINARY,
    fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL
)

agent = (AgentBuilder
    .for_vllm("microsoft/DialoGPT-medium")
    .with_gasa(gasa_config)
    .build())

# For OpenAI
gasa_config = GASAConfig.for_openai()
agent = (AgentBuilder
    .for_openai("gpt-4o")
    .with_gasa(gasa_config)
    .build())
```

### Direct GASA Service Usage

```python
from saplings.gasa import GASAService, GASAConfig
from saplings.memory import DependencyGraph, Document

# Create documents and graph
documents = [
    Document(id="doc1", content="Introduction to AI"),
    Document(id="doc2", content="Machine Learning basics"),
    Document(id="doc3", content="Deep Learning concepts")
]

graph = DependencyGraph()
for doc in documents:
    graph.add_document_node(doc)
graph.add_relationship("doc1", "doc2", "relates_to", 0.8)
graph.add_relationship("doc2", "doc3", "builds_upon", 0.9)

# Create GASA service
config = GASAConfig.default()
gasa_service = GASAService(graph=graph, config=config)

# Apply GASA to a prompt
prompt = "Summarize the key concepts in AI and machine learning"
result = gasa_service.apply_gasa(
    documents=documents,
    prompt=prompt,
    model_supports_sparse_attention=True  # For vLLM models
)

# Result contains modified prompt/masks for the model
print(f"Attention mask shape: {result['attention_mask'].shape}")
```

### Building Attention Masks

```python
from saplings.gasa import MaskFormat, MaskType

# Build different types of masks
dense_mask = gasa_service.build_mask(
    documents=documents,
    prompt=prompt,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION
)

sparse_mask = gasa_service.build_mask(
    documents=documents,
    prompt=prompt,
    format=MaskFormat.SPARSE,
    mask_type=MaskType.ATTENTION
)
```

## Fallback Strategies

### Block-Diagonal Packing

For models that don't support sparse attention masks, GASA can reorder tokens to create a block-diagonal structure:

```python
config = GASAConfig(
    fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
    block_size=512,
    overlap=64
)

# Apply block-diagonal packing
result = gasa_service.apply_gasa(
    documents=documents,
    prompt=prompt,
    input_ids=input_ids,
    model_supports_sparse_attention=False
)

# Returns reordered input_ids and position mapping
reordered_ids = result['input_ids']
position_mapping = result['position_mapping']
```

### Prompt Composition

For third-party APIs, GASA can restructure prompts based on graph relationships:

```python
config = GASAConfig(
    fallback_strategy=FallbackStrategy.PROMPT_COMPOSER,
    enable_prompt_composer=True,
    focus_tags=True,
    core_tag="[CORE_CTX]",
    near_tag="[NEAR_CTX]",
    summary_tag="[SUMMARY_CTX]"
)

composed_prompt = gasa_service.compose_prompt(
    documents=documents,
    prompt=prompt,
    system_prompt="You are a helpful AI assistant"
)
```

### Shadow Model Strategy

Use a lightweight local model for tokenization and mask generation:

```python
config = GASAConfig(
    fallback_strategy=FallbackStrategy.SHADOW_MODEL,
    enable_shadow_model=True,
    shadow_model_name="Qwen/Qwen3-0.6B",
    shadow_model_device="cpu"
)
```

## Advanced Features

### Visualization

```python
from saplings.gasa import MaskVisualizer

config = GASAConfig(
    visualize=True,
    visualization_dir="./gasa_visualizations"
)

# Visualizations are automatically saved when masks are built
visualizer = MaskVisualizer()
visualizer.visualize_mask(
    mask=attention_mask,
    output_path="./attention_heatmap.png",
    title="GASA Attention Pattern"
)
```

### Graph Distance Calculation

```python
from saplings.gasa import GraphDistanceCalculator

calculator = GraphDistanceCalculator(graph)
distance = calculator.get_distance(
    source_id="doc1",
    target_id="doc3",
    max_hops=3
)
print(f"Distance between doc1 and doc3: {distance} hops")
```

### Token Mapping

```python
from saplings.gasa import TokenMapper

mapper = TokenMapper(tokenizer)
token_info = mapper.map_tokens_to_documents(
    documents=documents,
    prompt=prompt
)
```

### Block Packing Utilities

```python
from saplings.gasa import block_pack

# Optimize token arrangement for block-diagonal attention
packed_tokens = block_pack(
    tokens=input_tokens,
    graph=dependency_graph,
    block_size=512,
    overlap=64
)
```

## Performance Optimization

### Caching Strategies

```python
# Enable mask caching for repeated queries
config = GASAConfig(
    cache_masks=True,
    cache_dir="./gasa_cache"
)

# Clear cache when needed
gasa_service.clear_cache()
```

### Memory Management

```python
# Configure for memory-constrained environments
config = GASAConfig(
    block_size=256,  # Smaller blocks
    overlap=32,      # Reduced overlap
    cache_masks=False,  # Disable caching
    visualize=False     # Disable visualization
)
```

### Optimization for Different Scenarios

```python
# High-performance configuration
high_perf_config = GASAConfig(
    max_hops=1,  # Aggressive pruning
    mask_strategy=MaskStrategy.BINARY,  # Fastest computation
    cache_masks=True,
    block_size=1024  # Larger blocks for efficiency
)

# Balanced configuration
balanced_config = GASAConfig(
    max_hops=2,
    mask_strategy=MaskStrategy.SOFT,
    soft_mask_temperature=0.1,
    cache_masks=True
)

# Quality-focused configuration
quality_config = GASAConfig(
    max_hops=3,  # More context
    mask_strategy=MaskStrategy.SOFT,
    soft_mask_temperature=0.05,  # Sharper gradients
    visualize=True  # Monitor attention patterns
)
```

## Examples

### Research Assistant with GASA

```python
from saplings import AgentBuilder
from saplings.gasa import GASAConfig, MaskStrategy, FallbackStrategy
from saplings.memory import Document, DependencyGraph

# Create research documents
papers = [
    Document(id="intro", content="Introduction to transformer architectures..."),
    Document(id="attention", content="Attention mechanisms in neural networks..."),
    Document(id="sparse", content="Sparse attention for efficiency...")
]

# Build dependency graph
graph = DependencyGraph()
for paper in papers:
    graph.add_document_node(paper)
graph.add_relationship("intro", "attention", "references", 0.9)
graph.add_relationship("attention", "sparse", "builds_upon", 0.8)

# Configure GASA for research
gasa_config = GASAConfig(
    enabled=True,
    max_hops=2,
    mask_strategy=MaskStrategy.SOFT,
    visualize=True,
    visualization_dir="./research_attention"
)

# Create research assistant
assistant = (AgentBuilder
    .for_openai("gpt-4o")
    .with_documents(papers)
    .with_graph(graph)
    .with_gasa(gasa_config)
    .build())

# Research query with GASA optimization
response = assistant.execute("Summarize the evolution of attention mechanisms in transformers")
```

### Code Analysis with Block-Diagonal Packing

```python
from saplings import AgentBuilder
from saplings.gasa import GASAConfig, FallbackStrategy

# Code files as documents
code_files = [
    Document(id="models.py", content=open("models.py").read()),
    Document(id="utils.py", content=open("utils.py").read()),
    Document(id="train.py", content=open("train.py").read())
]

# GASA config for local model
gasa_config = GASAConfig(
    enabled=True,
    max_hops=2,
    fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
    block_size=512,
    overlap=64
)

# Code analyzer with GASA
analyzer = (AgentBuilder
    .for_vllm("codellama/CodeLlama-7b-Python-hf")
    .with_documents(code_files)
    .with_gasa(gasa_config)
    .build())

analysis = analyzer.execute("Identify potential bugs and optimization opportunities")
```

### Multi-Modal Processing with GASA

```python
# Mixed content documents
documents = [
    Document(id="text", content="Market analysis report..."),
    Document(id="data", content="CSV data: sales, revenue, growth..."),
    Document(id="chart", content="Chart description: quarterly trends...")
]

# GASA with shadow model for API usage
gasa_config = GASAConfig(
    enabled=True,
    max_hops=2,
    fallback_strategy=FallbackStrategy.SHADOW_MODEL,
    enable_shadow_model=True,
    shadow_model_name="Qwen/Qwen3-0.6B"
)

agent = (AgentBuilder
    .for_anthropic("claude-3-sonnet-20240229")
    .with_documents(documents)
    .with_gasa(gasa_config)
    .build())
```

## Troubleshooting

### Common Issues

#### GASA Not Applying

```python
# Check if GASA is enabled
if not gasa_service.config.enabled:
    print("GASA is disabled")

# Verify graph connectivity
if not graph.has_path("doc1", "doc2"):
    print("Documents are not connected in graph")

# Check fallback strategy compatibility
if config.fallback_strategy == FallbackStrategy.BLOCK_DIAGONAL and input_ids is None:
    print("Block diagonal requires input_ids")
```

#### Performance Issues

```python
# Enable profiling
import cProfile
cProfile.run('gasa_service.build_mask(...)')

# Reduce complexity
config.max_hops = 1  # Reduce from 2 or 3
config.cache_masks = True  # Enable caching
config.visualize = False  # Disable visualization
```

#### Memory Issues

```python
# Reduce memory usage
config.block_size = 256  # Smaller blocks
config.cache_masks = False  # Disable caching
config.soft_mask_temperature = 1.0  # Less precise masks
```

### Debugging Tools

```python
# Check mask statistics
mask = gasa_service.build_mask(documents, prompt)
sparsity = 1.0 - (np.count_nonzero(mask) / mask.size)
print(f"Mask sparsity: {sparsity:.2%}")

# Visualize attention patterns
if config.visualize:
    # Check visualization directory
    import os
    viz_files = os.listdir(config.visualization_dir)
    print(f"Generated visualizations: {viz_files}")

# Monitor graph distances
calculator = GraphDistanceCalculator(graph)
for doc1 in documents:
    for doc2 in documents:
        dist = calculator.get_distance(doc1.id, doc2.id, config.max_hops)
        print(f"Distance {doc1.id} -> {doc2.id}: {dist}")
```

### Performance Benchmarking

```python
import time
from saplings.gasa import GASAService

# Benchmark mask generation
start_time = time.time()
mask = gasa_service.build_mask(documents, prompt)
mask_time = time.time() - start_time

# Benchmark apply_gasa
start_time = time.time()
result = gasa_service.apply_gasa(documents, prompt, model_supports_sparse_attention=True)
apply_time = time.time() - start_time

print(f"Mask generation: {mask_time:.3f}s")
print(f"Apply GASA: {apply_time:.3f}s")
print(f"Mask sparsity: {1 - np.count_nonzero(mask) / mask.size:.2%}")
```

This guide covers the complete GASA functionality in Saplings. GASA is a sophisticated attention optimization system that provides significant computational benefits while maintaining or improving model performance through graph-aware attention patterns.