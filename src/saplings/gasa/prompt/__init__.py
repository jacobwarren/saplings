from __future__ import annotations

"""
Prompt module for Graph-Aligned Sparse Attention (GASA).

This module provides prompt composition strategies for GASA, particularly
for models that don't support custom attention masks.
"""


from saplings.gasa.prompt.prompt_composer import GASAPromptComposer

__all__ = [
    "GASAPromptComposer",
]
