from __future__ import annotations

"""
Resilience module for core components.

This module provides resilience functionality for the Saplings framework.
"""

from saplings.core._internal.resilience.resilience import (
    DEFAULT_TIMEOUT,
    get_executor,
    retry,
    run_in_executor,
    with_timeout,
)
from saplings.core._internal.resilience.resilience_patterns import (
    ResiliencePatterns,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "run_in_executor",
    "with_timeout",
    "retry",
    "get_executor",
    "ResiliencePatterns",
]
