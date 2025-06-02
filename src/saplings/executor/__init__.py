from __future__ import annotations

"""
Executor module for Saplings.

This module re-exports the public API from saplings.api.executor.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides the executor functionality for Saplings, including:
- Speculative draft generation with low temperature
- Streaming output capabilities
- GASA mask injection mechanism
- Integration with JudgeAgent for verification
- Refinement logic for rejected outputs
- Performance optimizations for latency reduction
"""

# Import from the public API
from saplings.api.executor import (
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
