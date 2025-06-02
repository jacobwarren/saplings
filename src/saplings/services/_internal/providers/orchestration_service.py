from __future__ import annotations

"""
saplings.services.orchestration_service.
=====================================

Encapsulates orchestration functionality:
- Graph execution management
- Multi-step workflow coordination
"""


import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from saplings.api.orchestration import GraphRunnerConfig
from saplings.core.resilience import DEFAULT_TIMEOUT, with_timeout
from saplings.orchestration._internal.graph_runner import GraphRunner

if TYPE_CHECKING:
    # Use Any for trace manager since the interface might not exist yet
    from typing import Any as ITraceManager

    from saplings.api.models import LLM

logger = logging.getLogger(__name__)


class OrchestrationService:
    """Service that manages workflow orchestration."""

    def __init__(
        self,
        model: Optional["LLM"] = None,  # Can be None for lazy initialization
        trace_manager: Optional["ITraceManager"] = None,
        config: Optional[GraphRunnerConfig] = None,
    ) -> None:
        """
        Initialize the orchestration service.

        Args:
        ----
            model: LLM model to use for orchestration (can be provided later)
            trace_manager: Optional trace manager for monitoring
            config: Optional configuration for the graph runner

        Note:
        ----
            This service supports lazy initialization. The model can be provided
            later when the service is first used.

        """
        self._model = model
        self._trace_manager = trace_manager
        self._config = config
        self._graph_runner: Optional[GraphRunner] = None
        self._initialized = False

        logger.info("OrchestrationService created (lazy initialization)")

    def _ensure_initialized(self) -> None:
        """Ensure the service is initialized on-demand."""
        if not self._initialized:
            # Check if we have a model
            if self._model is None:
                raise ValueError("Model must be provided before using OrchestrationService")

            # Create graph runner config if not provided
            graph_runner_config = self._config or GraphRunnerConfig()

            # Configure monitoring if trace manager is available
            if self._trace_manager:
                graph_runner_config.enable_monitoring = True
                graph_runner_config.trace_manager = self._trace_manager

            # Create graph runner
            self._graph_runner = GraphRunner(
                model=self._model,
                config=graph_runner_config,
            )

            self._initialized = True
            logger.info("OrchestrationService initialized on-demand")

    @property
    def graph_runner(self) -> GraphRunner:
        """Get the underlying graph runner, initializing if necessary."""
        self._ensure_initialized()
        assert self._graph_runner is not None, "Graph runner should be initialized"
        return self._graph_runner

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
        # Ensure the service is initialized
        self._ensure_initialized()

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
    def inner_graph_runner(self) -> GraphRunner:
        """Get the underlying graph runner, initializing if necessary."""
        return self.graph_runner
