from __future__ import annotations

"""
Execution Service API module for Saplings.

This module provides the execution service implementation.
"""

from saplings.api.stability import stable
from saplings.services._internal.providers.execution_service import (
    ExecutionService as _ExecutionService,
)


@stable
class ExecutionService(_ExecutionService):
    """
    Service for executing tasks.

    This service provides functionality for executing tasks, including
    handling context, tools, and validation.
    """


__all__ = [
    "ExecutionService",
]
