from __future__ import annotations

"""
Public API for the Executor module.

This module provides the public API for the Executor module, which is responsible
for generating text with speculative execution, GASA mask injection, and verification.
"""

from saplings.executor._internal import (
    ExecutionResult,
    Executor,
    ExecutorConfig,
    RefinementStrategy,
    ValidationStrategy,
)

__all__ = [
    "ExecutionResult",
    "Executor",
    "ExecutorConfig",
    "RefinementStrategy",
    "ValidationStrategy",
]
