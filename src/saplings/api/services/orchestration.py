from __future__ import annotations

"""
Orchestration Service API module for Saplings.

This module provides the orchestration service implementation.
"""

from saplings.api.stability import stable
from saplings.services._internal.providers.orchestration_service import (
    OrchestrationService as _OrchestrationService,
)


@stable
class OrchestrationService(_OrchestrationService):
    """
    Service for orchestrating the agent workflow.

    This service provides functionality for orchestrating the agent workflow,
    including retrieval, planning, execution, and validation.
    """


__all__ = [
    "OrchestrationService",
]
