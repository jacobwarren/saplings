"""
Executor module for Saplings.

This module provides the executor functionality for Saplings, including:
- Speculative draft generation with low temperature
- Streaming output capabilities
- GASA mask injection mechanism
- Integration with JudgeAgent for verification
- Refinement logic for rejected outputs
- Performance optimizations for latency reduction
"""

from saplings.executor.config import ExecutorConfig, RefinementStrategy, VerificationStrategy
from saplings.executor.executor import ExecutionResult, Executor

__all__ = [
    "Executor",
    "ExecutionResult",
    "ExecutorConfig",
    "RefinementStrategy",
    "VerificationStrategy",
]
