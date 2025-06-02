from __future__ import annotations

"""
Model type definitions for Saplings.

This module provides model-related type definitions used across the Saplings framework.
"""

from typing import Any, Dict

# Model-related types
ModelConfig = Dict[str, Any]
ModelResponse = Dict[str, Any]

__all__ = [
    "ModelConfig",
    "ModelResponse",
]
