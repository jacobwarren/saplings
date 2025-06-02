from __future__ import annotations

"""
Memory Service API module for Saplings.

This module provides the memory service implementation.
"""

from saplings.api.stability import stable
from saplings.services._internal.managers.memory_manager import MemoryManager as _MemoryManager


@stable
class MemoryManager(_MemoryManager):
    """
    Service for managing memory.

    This service provides functionality for managing memory, including
    adding documents, retrieving documents, and managing the dependency graph.
    """


__all__ = [
    "MemoryManager",
]
