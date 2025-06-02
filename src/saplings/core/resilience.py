from __future__ import annotations

"""
Resilience module for Saplings.

This module provides utilities for making code more resilient to failures.
"""

from saplings.core._internal.exceptions import (
    CircuitBreakerError,
    OperationCancelledError,
    OperationTimeoutError,
)
from saplings.core._internal.resilience.resilience import (
    DEFAULT_TIMEOUT,
    CircuitBreaker,
    Validation,
    get_executor,
    retry,
    run_in_executor,
    with_timeout,
    with_timeout_decorator,
)
from saplings.core._internal.resilience.resilience_patterns import (
    ResiliencePatterns,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "CircuitBreaker",
    "CircuitBreakerError",
    "OperationCancelledError",
    "OperationTimeoutError",
    "ResiliencePatterns",
    "Validation",
    "get_executor",
    "retry",
    "run_in_executor",
    "with_timeout",
    "with_timeout_decorator",
]
