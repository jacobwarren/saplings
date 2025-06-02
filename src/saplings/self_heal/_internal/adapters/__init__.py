from __future__ import annotations

"""
Adapters module for self-healing components.

This module provides adapter functionality for the Saplings framework.
"""

from saplings.self_heal._internal.adapters.adapter_manager import (
    Adapter,
    AdapterManager,
    AdapterMetadata,
    AdapterPriority,
)

__all__ = [
    "Adapter",
    "AdapterManager",
    "AdapterMetadata",
    "AdapterPriority",
]
