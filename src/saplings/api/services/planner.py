from __future__ import annotations

"""
Planner Service API module for Saplings.

This module provides the planner service implementation.
"""

from saplings.api.stability import stable
from saplings.services._internal.providers.planner_service import PlannerService as _PlannerService


@stable
class PlannerService(_PlannerService):
    """
    Service for planning tasks.

    This service provides functionality for planning tasks, including
    breaking down tasks into steps and allocating budget.
    """


__all__ = [
    "PlannerService",
]
