from __future__ import annotations

"""
Test-specific trace viewer for Saplings.

This module provides a specialized TraceViewer implementation for testing,
with additional methods to access plan steps and model calls for assertions.
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.monitoring.trace_viewer import TraceViewer

if TYPE_CHECKING:
    from saplings.monitoring.config import MonitoringConfig
    from saplings.monitoring.trace import TraceManager
    from saplings.planner.plan_step import PlanStep

logger = logging.getLogger(__name__)


class TestTraceViewer(TraceViewer):
    """
    Test-specific trace viewer with additional methods for testing.

    This class extends the standard TraceViewer with methods to access
    plan steps and model calls for assertions in tests.
    """

    def __init__(
        self,
        trace_manager: TraceManager | None = None,
        config: MonitoringConfig | None = None,
    ) -> None:
        """
        Initialize the TestTraceViewer.

        Args:
        ----
            trace_manager: Trace manager to use
            config: Monitoring configuration

        """
        super().__init__(trace_manager, config)
        self._last_plan = None
        self._last_model_calls = []

    @property
    def last_plan(self):
        """
        Get the last plan created.

        Returns
        -------
            List[PlanStep]: The last plan created, or an empty list if none

        """
        return self._last_plan or []

    @last_plan.setter
    def last_plan(self, plan: list[PlanStep]) -> None:
        """
        Set the last plan created.

        Args:
        ----
            plan: The plan to set

        """
        self._last_plan = plan

    @property
    def last_model_calls(self):
        """
        Get the last model calls made.

        Returns
        -------
            List[Dict[str, Any]]: The last model calls made, or an empty list if none

        """
        return self._last_model_calls

    @last_model_calls.setter
    def last_model_calls(self, calls: list[dict[str, Any]]) -> None:
        """
        Set the last model calls made.

        Args:
        ----
            calls: The model calls to set

        """
        self._last_model_calls = calls

    def record_plan(self, plan: list[PlanStep]) -> None:
        """
        Record a plan for testing.

        Args:
        ----
            plan: The plan to record

        """
        self._last_plan = plan

    def record_model_call(
        self, prompt: str, response: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Record a model call for testing.

        Args:
        ----
            prompt: The prompt sent to the model
            response: The response from the model
            metadata: Additional metadata about the call

        """
        call = {"prompt": prompt, "response": response, "metadata": metadata or {}}
        self._last_model_calls.append(call)

    def visualize_trace(self, trace_id: str, output_path: str | None = None) -> dict[str, Any]:
        """
        Visualize a trace (stub implementation for testing).

        Args:
        ----
            trace_id: ID of the trace to visualize
            output_path: Path to save the visualization

        Returns:
        -------
            Dict[str, Any]: Visualization data

        """
        # This is a stub implementation for testing
        return {"trace_id": trace_id, "output_path": output_path}
