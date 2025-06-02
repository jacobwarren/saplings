from __future__ import annotations

"""
saplings.services.orchestration_service.
=====================================

Encapsulates orchestration functionality:
- Graph execution management
- Multi-step workflow coordination
"""


import logging
from typing import TYPE_CHECKING, Any, Dict

from saplings.core.resilience import DEFAULT_TIMEOUT, with_timeout
from saplings.orchestration.config import GraphRunnerConfig
from saplings.orchestration.graph_runner import GraphRunner

if TYPE_CHECKING:
    from saplings.core.model_adapter import LLM

# Optional dependency (monitoring)
try:
    from saplings.monitoring.trace import TraceManager
except ModuleNotFoundError:  # pragma: no cover
    TraceManager = None  # type: ignore

logger = logging.getLogger(__name__)


class OrchestrationService:
    """Service that manages workflow orchestration."""

    def __init__(
        self,
        model: "LLM",
        trace_manager: Any = None,
    ) -> None:
        self._trace_manager = trace_manager

        # Create graph runner config
        graph_runner_config = GraphRunnerConfig()

        # Create graph runner
        self.graph_runner = GraphRunner(
            model=model,
            config=graph_runner_config,
        )

        logger.info("OrchestrationService initialized")

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    async def run_workflow(
        self,
        workflow_definition: Dict[str, Any],
        inputs: Dict[str, Any],
        trace_id: str | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Run a workflow defined as a graph.

        Args:
        ----
            workflow_definition: The workflow graph definition
            inputs: Initial inputs to the workflow
            trace_id: Optional trace ID for monitoring
            timeout: Optional timeout in seconds

        Returns:
        -------
            Workflow execution results

        Raises:
        ------
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        span = None
        if self._trace_manager:
            span = self._trace_manager.start_span(
                name="OrchestrationService.run_workflow",
                trace_id=trace_id,
                attributes={"component": "orchestration_service"},
            )

        try:
            # Define workflow execution function
            async def _run_workflow():
                # Since GraphRunner doesn't have a run method, we'll use negotiate as a fallback
                # This is a temporary solution until the proper run method is implemented
                result = await self.graph_runner.negotiate(
                    task=str(workflow_definition),
                    context=str(inputs),
                    max_rounds=10,
                    timeout_seconds=int(timeout) if timeout else None,
                )
                # Convert string result to a dictionary to match the expected return type
                return {"result": result}

            # Execute with timeout
            return await with_timeout(
                _run_workflow(), timeout=timeout, operation_name="run_workflow"
            )
        except Exception as e:
            logger.exception(f"Error running workflow: {e}")
            raise
        finally:
            if span and self._trace_manager:
                self._trace_manager.end_span(span.span_id)

    # For advanced use cases or backward compatibility
    @property
    def inner_graph_runner(self):
        """Get the underlying graph runner."""
        return self.graph_runner
