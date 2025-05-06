from __future__ import annotations

"""
Orchestration service interface for Saplings.

This module defines the interface for orchestration operations that manage
workflows and multi-agent coordination. This is a pure interface with no
implementation details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any


class IOrchestrationService(ABC):
    """Interface for orchestration operations."""

    @abstractmethod
    async def orchestrate(
        self,
        workflow: dict[str, Any],
        context: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Orchestrate a workflow.

        Args:
        ----
            workflow: Workflow definition
            context: Optional workflow context
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dict[str, Any]: Orchestration results

        """

    @abstractmethod
    async def register_agent(
        self, agent_id: str, agent_config: dict[str, Any], trace_id: str | None = None
    ) -> bool:
        """
        Register an agent with the orchestrator.

        Args:
        ----
            agent_id: Agent identifier
            agent_config: Agent configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            bool: Whether registration was successful

        """

    @abstractmethod
    async def create_workflow(
        self, workflow_type: str, steps: list[dict[str, Any]], trace_id: str | None = None
    ) -> dict[str, Any]:
        """
        Create a workflow.

        Args:
        ----
            workflow_type: Type of workflow
            steps: Workflow steps
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dict[str, Any]: Created workflow

        """

    @abstractmethod
    async def get_workflow_status(
        self, workflow_id: str, trace_id: str | None = None
    ) -> dict[str, Any]:
        """
        Get the status of a workflow.

        Args:
        ----
            workflow_id: Workflow identifier
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Dict[str, Any]: Workflow status

        """
