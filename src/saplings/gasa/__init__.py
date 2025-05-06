from __future__ import annotations

"""
Graph-Aligned Sparse Attention (GASA) module for Saplings.

This module provides the GASA functionality for Saplings, including:
- Graph traversal algorithm for token-level mask generation
- Mask serialization/deserialization
- Integration points for executor.py
- Block-diagonal packing fallback for unsupported models
- Visualization utilities for debugging masks
- Shadow model tokenization for third-party LLM APIs
- Graph-aware prompt composition for third-party LLM APIs

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


from saplings.gasa.builder.standard_mask_builder import StandardMaskBuilder
from saplings.gasa.config import FallbackStrategy, GASAConfig, MaskStrategy
from saplings.gasa.core.types import MaskFormat, MaskType
from saplings.gasa.packing.block_diagonal_packer import BlockDiagonalPacker

# For backward compatibility
from saplings.gasa.packing.block_diagonal_packer import (
    BlockDiagonalPacker as GASABlockDiagonalPacker,
)
from saplings.gasa.prompt.prompt_composer import GASAPromptComposer
from saplings.gasa.service.gasa_service import GASAService
from saplings.gasa.visualization.mask_visualizer import MaskVisualizer

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
    "GASAPromptComposer",
    "GASAService",
    "MaskFormat",
    "MaskStrategy",
    "MaskType",
    "MaskVisualizer",
    "StandardMaskBuilder",
]

# Add shadow model tokenizer to exports if available
if SHADOW_MODEL_AVAILABLE:
    __all__.append("ShadowModelTokenizer")
