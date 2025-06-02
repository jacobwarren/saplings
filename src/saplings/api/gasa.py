from __future__ import annotations

"""
Public API for Graph-Aligned Sparse Attention (GASA).

This module provides the public API for GASA functionality, including:
- Configuration for GASA
- Block diagonal packing for third-party LLMs
- Visualization utilities for GASA masks
- Graph distance calculation
- GASA service for centralized management
- Token mapping for document-token relationships
- Block packing utilities for optimized performance
"""

from typing import Any, Protocol, runtime_checkable

from saplings.api.stability import beta, stable

# Import from internal modules
from saplings.gasa._internal.builder.standard_mask_builder import StandardMaskBuilder
from saplings.gasa._internal.builder.token_tracking_mask_builder import TokenTrackingMaskBuilder
from saplings.gasa._internal.config import FallbackStrategy, GASAConfig, MaskStrategy
from saplings.gasa._internal.core.graph_distance import GraphDistanceCalculator
from saplings.gasa._internal.core.token_mapper import TokenMapper
from saplings.gasa._internal.core.types import MaskFormat, MaskType
from saplings.gasa._internal.gasa_config_builder import GASAConfigBuilder
from saplings.gasa._internal.packing.block_diagonal_packer import BlockDiagonalPacker
from saplings.gasa._internal.packing.block_pack import block_pack
from saplings.gasa._internal.prompt.prompt_composer import GASAPromptComposer
from saplings.gasa._internal.service.gasa_service import GASAService
from saplings.gasa._internal.service.gasa_service_builder import GASAServiceBuilder
from saplings.gasa._internal.service.null_gasa_service import NullGASAService
from saplings.gasa._internal.visualization.mask_visualizer import MaskVisualizer


@runtime_checkable
@stable
class Document(Protocol):
    """Protocol for document objects."""

    id: str
    content: str
    metadata: dict[str, Any] | None

    def __getattr__(self, name: str) -> Any: ...


@runtime_checkable
@stable
class DependencyGraph(Protocol):
    """Protocol for dependency graph objects."""

    def get_neighbors(self, node_id: str) -> list[str]: ...
    def get_distance(self, source_id: str, target_id: str) -> int | float: ...
    def get_subgraph(self, node_ids: list[str], max_hops: int = 2) -> "DependencyGraph": ...
    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None: ...
    def add_edge(
        self, source_id: str, target_id: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...


# Re-export for public API
__all__ = [
    # Configuration
    "GASAConfig",
    "GASAConfigBuilder",
    "FallbackStrategy",
    "MaskStrategy",
    "MaskFormat",
    "MaskType",
    # Core functionality
    "BlockDiagonalPacker",
    "StandardMaskBuilder",
    "TokenTrackingMaskBuilder",
    "GraphDistanceCalculator",
    "TokenMapper",
    "GASAService",
    "GASAServiceBuilder",
    "NullGASAService",
    "block_pack",
    "GASAPromptComposer",
    # Visualization
    "MaskVisualizer",
    # Protocols
    "Document",
    "DependencyGraph",
]

# Add stability annotations to re-exported classes
GASAConfig = beta(GASAConfig)
GASAConfigBuilder = beta(GASAConfigBuilder)
FallbackStrategy = beta(FallbackStrategy)
MaskStrategy = beta(MaskStrategy)
MaskFormat = beta(MaskFormat)
MaskType = beta(MaskType)
BlockDiagonalPacker = beta(BlockDiagonalPacker)
StandardMaskBuilder = beta(StandardMaskBuilder)
TokenTrackingMaskBuilder = beta(TokenTrackingMaskBuilder)
GraphDistanceCalculator = beta(GraphDistanceCalculator)
TokenMapper = beta(TokenMapper)
GASAService = beta(GASAService)
GASAServiceBuilder = beta(GASAServiceBuilder)
NullGASAService = beta(NullGASAService)
block_pack = beta(block_pack)
GASAPromptComposer = beta(GASAPromptComposer)
MaskVisualizer = beta(MaskVisualizer)

# Add docstrings to re-exported classes for better API documentation

# Update docstring for GASAConfig
GASAConfig.__doc__ = """
Configuration for Graph-Aligned Sparse Attention (GASA).

This class provides configuration options for GASA, including:
- Whether GASA is enabled
- Maximum number of hops for attention
- Mask strategy (binary or soft)
- Fallback strategy for models that don't support sparse attention
- Visualization options
- Shadow model configuration for third-party LLMs
- Prompt composer configuration for third-party LLMs

Basic Example:
```python
# Create a GASA config
config = GASAConfig(
    enabled=True,
    max_hops=2,
    mask_strategy=MaskStrategy.BINARY,
    fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
    visualize=True,
    visualization_dir="./gasa_visualizations",
)
```

Advanced Example with All Options:
```python
# Create a GASA config with all options
config = GASAConfig(
    # Basic settings
    enabled=True,
    max_hops=2,
    mask_strategy=MaskStrategy.BINARY,
    fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,

    # Global token settings
    global_tokens=["[CLS]", "[SEP]", "<s>", "</s>", "[SUM]"],
    summary_token="[SUM]",
    add_summary_token=True,

    # Block diagonal packing settings
    block_size=512,
    overlap=64,

    # Soft mask settings
    soft_mask_temperature=0.1,

    # Caching settings
    cache_masks=True,
    cache_dir="./gasa_cache",

    # Visualization settings
    visualize=True,
    visualization_dir="./gasa_visualizations",

    # Shadow model settings for third-party LLMs
    enable_shadow_model=True,
    shadow_model_name="Qwen/Qwen3-1.8B",
    shadow_model_device="cpu",
    shadow_model_cache_dir="./shadow_model_cache",

    # Prompt composer settings for third-party LLMs
    enable_prompt_composer=True,
    focus_tags=True,
    core_tag="[CORE_CTX]",
    near_tag="[NEAR_CTX]",
    summary_tag="[SUMMARY_CTX]",
)
```

Provider-Specific Configurations:
```python
# Configuration optimized for OpenAI
config = GASAConfig(
    enabled=True,
    max_hops=2,
    fallback_strategy=FallbackStrategy.PROMPT_COMPOSER,
    enable_shadow_model=True,
    shadow_model_name="Qwen/Qwen3-0.6B",
    enable_prompt_composer=True,
    focus_tags=True,
)

# Configuration optimized for Anthropic
config = GASAConfig(
    enabled=True,
    max_hops=2,
    fallback_strategy=FallbackStrategy.PROMPT_COMPOSER,
    enable_shadow_model=True,
    shadow_model_name="Qwen/Qwen3-0.6B",
    enable_prompt_composer=True,
    focus_tags=True,
)

# Configuration optimized for vLLM
config = GASAConfig(
    enabled=True,
    max_hops=2,
    fallback_strategy=FallbackStrategy.BLOCK_DIAGONAL,
    enable_shadow_model=False,
    enable_prompt_composer=False,
)
```
"""

# Update docstring for BlockDiagonalPacker
BlockDiagonalPacker.__doc__ = """
Block diagonal packer for GASA.

This class provides functionality for reordering tokens to create a block-diagonal
structure for models that don't support sparse attention.

Example:
```python
# Create a block diagonal packer
packer = BlockDiagonalPacker(
    graph=graph,
    config=gasa_config,
    tokenizer=tokenizer,
)

# Reorder tokens
result = packer.reorder_tokens(
    documents=documents,
    prompt=prompt,
    input_ids=input_ids,
)
```
"""

# Update docstring for MaskVisualizer
MaskVisualizer.__doc__ = """
Visualizer for GASA masks.

This class provides functionality for visualizing GASA masks for debugging and
understanding how GASA works.

Example:
```python
# Create a mask visualizer
visualizer = MaskVisualizer(config=gasa_config)

# Visualize a mask
visualizer.visualize_mask(
    mask=mask,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
    output_path="./gasa_mask.png",
    title="GASA Attention Mask",
    show=True,
)
```
"""

# Update docstring for StandardMaskBuilder
StandardMaskBuilder.__doc__ = """
Standard mask builder for GASA.

This class provides functionality for building GASA masks based on the dependency
graph and document relationships.

Example:
```python
# Create a standard mask builder
mask_builder = StandardMaskBuilder(
    graph=graph,
    config=gasa_config,
    tokenizer=tokenizer,
)

# Build a mask
mask = mask_builder.build_mask(
    documents=documents,
    prompt=prompt,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
)
```
"""

# Update docstring for GraphDistanceCalculator
GraphDistanceCalculator.__doc__ = """
Graph distance calculator for GASA.

This class provides functionality for calculating distances between nodes in the
dependency graph, which is used for building GASA masks.

Example:
```python
# Create a graph distance calculator
calculator = GraphDistanceCalculator(graph=graph)

# Calculate distances
distances = calculator.build_distance_matrix(
    node_ids=[doc.id for doc in documents],
    max_hops=2,
)
```
"""

# Update docstring for TokenMapper
TokenMapper.__doc__ = """
Token mapper for GASA.

This class provides functionality for mapping between tokens and document chunks,
which is used for building accurate GASA masks.

Example:
```python
# Create a token mapper
token_mapper = TokenMapper(tokenizer=tokenizer)

# Add a document chunk
token_mapper.add_document_chunk(
    chunk=document,
    document_id=document.id,
    node_id=f"document:{document.id}"
)
```
"""

# Update docstring for GASAService
GASAService.__doc__ = """
Service for managing Graph-Aligned Sparse Attention functionality.

This class serves as a central point of integration for GASA, managing
the creation and coordination of mask builders, packers, and prompt composers.
It also provides methods for applying GASA to prompts and inputs based on
model capabilities and configuration.

Example:
```python
# Create a GASA service
service = GASAService(
    graph=graph,
    config=gasa_config,
    tokenizer=tokenizer,
)

# Build a mask
mask = service.build_mask(
    documents=documents,
    prompt=prompt,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
)
```
"""

# Update docstring for GASAServiceBuilder
GASAServiceBuilder.__doc__ = """
Builder for GASAService.

This class provides a fluent interface for building GASAService instances with
proper configuration and dependency injection. It separates configuration
from initialization and ensures all required dependencies are provided.

Example:
```python
# Create a builder for GASAService
builder = GASAServiceBuilder()

# Configure the builder with dependencies and options
service = builder.with_graph(dependency_graph) \\
                .with_config(gasa_config) \\
                .with_tokenizer(tokenizer) \\
                .build()
```
"""

# Update docstring for GASAConfigBuilder
GASAConfigBuilder.__doc__ = """
Builder for GASAConfig.

This class provides a fluent interface for building GASAConfig instances with
proper configuration. It separates configuration from initialization and
provides a fluent interface for setting configuration parameters.

Example:
```python
# Create a builder for GASAConfig
builder = GASAConfigBuilder()

# Configure the builder with options
config = builder.with_enabled(True) \\
               .with_max_hops(2) \\
               .with_mask_strategy(MaskStrategy.BINARY) \\
               .with_fallback_strategy(FallbackStrategy.BLOCK_DIAGONAL) \\
               .build()
```
"""

# Update docstring for block_pack
block_pack.__doc__ = """
Optimized block packing function for GASA.

This function provides an optimized implementation of block packing using PyTorch,
which is significantly faster than the pure Python implementation.

Example:
```python
import torch

# Create some masks
masks = [torch.ones((10, 10)) for _ in range(5)]

# Pack the masks
result = block_pack(masks)
```
"""

# Update docstring for TokenTrackingMaskBuilder
TokenTrackingMaskBuilder.__doc__ = """
Enhanced mask builder that tracks tokens during prompt construction.

This class extends the StandardMaskBuilder with more robust token tracking
to ensure accurate mapping between tokens and document chunks.

Example:
```python
# Create a token tracking mask builder
mask_builder = TokenTrackingMaskBuilder(
    graph=graph,
    config=gasa_config,
    tokenizer=tokenizer,
)

# Build a mask
mask = mask_builder.build_mask(
    documents=documents,
    prompt=prompt,
    format=MaskFormat.DENSE,
    mask_type=MaskType.ATTENTION,
)
```
"""

# Update docstring for GASAPromptComposer
GASAPromptComposer.__doc__ = """
Graph-aware prompt composer for GASA with third-party LLM APIs.

This class structures prompts based on graph relationships for use with
third-party LLM APIs like OpenAI and Anthropic. It implements the block-diagonal
packing strategy and adds focus tags to important context.

Example:
```python
# Create a prompt composer
composer = GASAPromptComposer(
    graph=graph,
    config=gasa_config,
)

# Compose a prompt
composed_prompt = composer.compose_prompt(
    documents=documents,
    prompt="What are the key concepts in these documents?",
    system_prompt="You are a helpful assistant that summarizes documents.",
)
```

Provider-Specific Usage:
```python
# For OpenAI
config = GASAConfig(
    enabled=True,
    fallback_strategy=FallbackStrategy.PROMPT_COMPOSER,
    enable_prompt_composer=True,
    focus_tags=True,
)
composer = GASAPromptComposer(graph=graph, config=config)
composed_prompt = composer.compose_prompt(documents, prompt, system_prompt)

# For Anthropic
config = GASAConfig(
    enabled=True,
    fallback_strategy=FallbackStrategy.PROMPT_COMPOSER,
    enable_prompt_composer=True,
    focus_tags=True,
)
composer = GASAPromptComposer(graph=graph, config=config)
composed_prompt = composer.compose_prompt(documents, prompt, system_prompt)
```
"""
