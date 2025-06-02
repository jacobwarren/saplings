from __future__ import annotations

"""
Planner service interface for Saplings.

This module defines the interface for planning operations that create and manage
task execution plans. This is a pure interface with no implementation details
or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Forward references
Document = Any  # From saplings.memory.document
PlanStep = Any  # From saplings.planner.plan_step


@dataclass
class PlannerConfig:
    """Configuration for planning operations."""

    max_steps: int = 5
    detail_level: str = "medium"
    model_name: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class PlanningResult:
    """Result of a planning operation."""

    plan: List[Dict[str, Any]]
    reasoning: str
    estimated_steps: int
    metadata: Optional[Dict[str, Any]] = None


class IPlannerService(ABC):
    """Interface for planning operations."""

    @abstractmethod
    async def create_plan_steps(
        self, task: str, context: list[Document] | None = None, trace_id: str | None = None
    ) -> list[PlanStep]:
        """
        Create plan steps for a task.

        Args:
        ----
            task: Description of the task to plan for
            context: Optional contextual documents
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            List[PlanStep]: The created plan steps

        """

    @abstractmethod
    async def refine_plan(
        self, plan: list[PlanStep], feedback: str, trace_id: str | None = None
    ) -> list[PlanStep]:
        """
        Refine an existing plan based on feedback.

        Args:
        ----
            plan: Existing plan steps
            feedback: Feedback for refinement
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            List[PlanStep]: The refined plan steps

        """

    @abstractmethod
    async def get_next_step(
        self, plan: list[PlanStep], trace_id: str | None = None
    ) -> PlanStep | None:
        """
        Get the next executable step from a plan.

        Args:
        ----
            plan: Plan steps
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Optional[PlanStep]: The next step or None if all steps are complete

        """

    @abstractmethod
    async def update_step_status(
        self,
        plan: list[PlanStep],
        step_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> list[PlanStep]:
        """
        Update the status of a plan step.

        Args:
        ----
            plan: Plan steps
            step_id: ID of the step to update
            status: New status
            result: Optional result of the step
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            List[PlanStep]: The updated plan steps

        """

    @abstractmethod
    async def create_plan(
        self,
        task: str,
        context: Optional[str] = None,
        config: Optional[PlannerConfig] = None,
        trace_id: Optional[str] = None,
    ) -> PlanningResult:
        """
        Create a plan for a task.

        Args:
        ----
            task: Task description
            context: Optional context information
            config: Optional planner configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            PlanningResult: Result of the planning operation

        """
