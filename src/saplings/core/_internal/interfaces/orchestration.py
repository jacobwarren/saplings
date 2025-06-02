from __future__ import annotations

"""
Orchestration service interface for Saplings.

This module defines the interface for orchestration operations that manage
workflows and multi-agent coordination. This is a pure interface with no
implementation details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class OrchestrationConfig:
    """Configuration for orchestration operations."""

    max_steps: int = 10
    timeout: Optional[float] = None
    communication_strategy: str = "direct"
    negotiation_strategy: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Result of an orchestration operation."""

    success: bool
    output: Any
    steps_executed: int
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None


class IOrchestrationService(ABC):
    """Interface for orchestration operations."""

    @abstractmethod
    async def orchestrate_workflow(
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

    @abstractmethod
    async def orchestrate(
        self,
        agents: List[Any],
        task: str,
        config: Optional[OrchestrationConfig] = None,
        trace_id: Optional[str] = None,
    ) -> OrchestrationResult:
        """
        Orchestrate multiple agents to complete a task.

        Args:
        ----
            agents: List of agents to orchestrate
            task: Task description
            config: Optional orchestration configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            OrchestrationResult: Result of the orchestration operation

        """
