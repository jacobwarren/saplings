from __future__ import annotations

"""
Self-healing module for Saplings.

This module provides self-healing capabilities for Saplings, including:
- PatchGenerator for auto-fixing errors
- Retry mechanism with capping
- Success pair collection for training
- LoRA fine-tuning pipeline
- Adapter management for model improvements
- Integration with JudgeAgent feedback
"""

# Import directly from internal modules to avoid circular imports
from saplings.self_heal._internal.adapters import AdapterPriority
from saplings.self_heal._internal.config import RetryStrategy
from saplings.self_heal._internal.patches import PatchStatus

# Re-export symbols
__all__ = [
    # Enums that can be safely re-exported
    "AdapterPriority",
    "RetryStrategy",
    "PatchStatus",
    # Note: Other self-heal symbols should be imported from saplings.api.self_heal
]
