from __future__ import annotations

"""
Graph-Aligned Sparse Attention (GASA) module for Saplings.

This module re-exports the public API from saplings.api.gasa.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

GASA injects a binary attention mask—derived from the retrieval dependency graph—into
each transformer layer, permitting full attention only between tokens whose source chunks
are ≤ h hops apart in the graph, while routing others through a lightweight global summary token.

This approach reduces computational cost (up to 40% fewer FLOPs) and improves grounding
by focusing the model's attention on relevant context.

For third-party LLM APIs like OpenAI and Anthropic that don't expose low-level attention
mechanisms, GASA provides alternative approaches:
1. Shadow model tokenization - Uses a small local model for tokenization and mask generation
2. Graph-aware prompt composition - Structures prompts based on graph relationships
3. Block-diagonal packing - Reorders chunks to create a block-diagonal structure
"""

# Import from the public API
from saplings.api.gasa import (
    BlockDiagonalPacker,
    FallbackStrategy,
    GASAConfig,
    GASAConfigBuilder,
    GASAPromptComposer,
    GASAService,
    GASAServiceBuilder,
    GraphDistanceCalculator,
    MaskFormat,
    MaskStrategy,
    MaskType,
    MaskVisualizer,
    NullGASAService,
    StandardMaskBuilder,
    TokenMapper,
    TokenTrackingMaskBuilder,
    block_pack,
)

# For backward compatibility
from saplings.api.gasa import BlockDiagonalPacker as GASABlockDiagonalPacker

# Import shadow model tokenizer if available
try:
    from saplings.tokenizers.shadow_model_tokenizer import ShadowModelTokenizer

    SHADOW_MODEL_AVAILABLE = True
except ImportError:
    SHADOW_MODEL_AVAILABLE = False

__all__ = [
    "BlockDiagonalPacker",
    "FallbackStrategy",
    "GASABlockDiagonalPacker",
    "GASAConfig",
    "GASAConfigBuilder",
    "GASAPromptComposer",
    "GASAService",
    "GASAServiceBuilder",
    "GraphDistanceCalculator",
    "MaskFormat",
    "MaskStrategy",
    "MaskType",
    "MaskVisualizer",
    "NullGASAService",
    "StandardMaskBuilder",
    "TokenMapper",
    "TokenTrackingMaskBuilder",
    "block_pack",
]

# Add shadow model tokenizer to exports if available
if SHADOW_MODEL_AVAILABLE:
    __all__.append("ShadowModelTokenizer")
