from __future__ import annotations

"""
Success pair collector for Saplings.

This module provides the public API for collecting success pairs for self-healing.
"""

from saplings.self_heal._internal.success_pair_collector import (
    SuccessPairCollector as _SuccessPairCollector,
)

# Re-export the SuccessPairCollector class
SuccessPairCollector = _SuccessPairCollector

__all__ = ["SuccessPairCollector"]
