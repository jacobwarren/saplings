"""
Fallback service implementations for optional dependencies.

This module provides null/mock implementations of services that depend on optional
dependencies, allowing the system to gracefully degrade when those dependencies
are not available.

These implementations follow the same interfaces as the real services but provide
minimal functionality or no-op behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

# Configure logging
logger = logging.getLogger(__name__)


class NullGASAService:
    """
    Fallback GASA service that does nothing.

    This service is used when torch/transformers dependencies are not available.
    It implements the IGASAService interface but returns None for mask creation,
    effectively disabling GASA functionality.
    """

    def __init__(self):
        """Initialize null GASA service."""
        logger.info("Initialized NullGASAService - GASA functionality disabled")

    def create_mask(self, graph: Any, tokens: List[str]) -> Any:
        """
        Create a mask from a graph and tokens.

        Args:
        ----
            graph: Document graph (ignored)
            tokens: List of tokens (ignored)

        Returns:
        -------
            None (no mask, use standard attention)

        """
        logger.debug("NullGASAService.create_mask called - returning None (no mask)")
        return None  # No mask, use standard attention

    def configure(self, config: Any) -> None:
        """
        Configure the GASA service.

        Args:
        ----
            config: GASA configuration (ignored)

        """
        logger.debug("NullGASAService.configure called - no configuration needed")

    def is_available(self) -> bool:
        """
        Check if GASA service is available.

        Returns
        -------
            False (null service is never "available")

        """
        return False


class NullMonitoringService:
    """
    Fallback monitoring service that logs to console.

    This service is used when langsmith or other monitoring dependencies are not available.
    It implements the IMonitoringService interface but only provides basic console logging.
    """

    def __init__(self):
        """Initialize null monitoring service."""
        logger.info("Initialized NullMonitoringService - using console logging")

    def log_event(self, event: Any) -> None:
        """
        Log an event.

        Args:
        ----
            event: Monitoring event to log

        """
        event_type = getattr(event, "event_type", "unknown")
        event_data = getattr(event, "data", {})

        # Simple console logging instead of full monitoring
        print(f"Event: {event_type} - {event_data}")
        logger.debug(f"NullMonitoringService logged event: {event_type}")

    def start_trace(self, trace_id: str) -> None:
        """
        Start a trace.

        Args:
        ----
            trace_id: ID of the trace to start

        """
        logger.debug(f"NullMonitoringService.start_trace called with ID: {trace_id}")

    def end_trace(self, trace_id: str) -> None:
        """
        End a trace.

        Args:
        ----
            trace_id: ID of the trace to end

        """
        logger.debug(f"NullMonitoringService.end_trace called with ID: {trace_id}")

    def get_traces(self) -> List[Any]:
        """
        Get all traces.

        Returns
        -------
            Empty list (no traces stored)

        """
        logger.debug("NullMonitoringService.get_traces called - returning empty list")
        return []

    def is_available(self) -> bool:
        """
        Check if monitoring service is available.

        Returns
        -------
            False (null service provides minimal functionality)

        """
        return False


class NullSelfHealingService:
    """
    Fallback self-healing service that doesn't actually heal.

    This service implements the ISelfHealingService interface but doesn't perform
    any actual healing operations. It's used when self-healing dependencies are
    not available or when self-healing is disabled.
    """

    def __init__(self):
        """Initialize null self-healing service."""
        logger.info("Initialized NullSelfHealingService - self-healing disabled")

    def fix_error(self, error: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix an error.

        Args:
        ----
            error: Error to fix
            context: Context information

        Returns:
        -------
            Result indicating no fix was applied

        """
        error_str = str(error)
        logger.debug(f"NullSelfHealingService.fix_error called for: {error_str}")

        return {
            "error_id": f"null_error_{id(error)}",
            "fixed": False,
            "patch": None,
            "message": "Self-healing disabled - no fix applied",
            "context": context,
        }

    def collect_success_pair(self, input_data: Any, output_data: Any) -> None:
        """
        Collect a success pair for learning.

        Args:
        ----
            input_data: Input that led to success
            output_data: Successful output

        """
        logger.debug("NullSelfHealingService.collect_success_pair called - no collection performed")

    def is_available(self) -> bool:
        """
        Check if self-healing service is available.

        Returns
        -------
            False (null service doesn't provide healing)

        """
        return False


class NullOrchestrationService:
    """
    Fallback orchestration service that skips complex orchestration.

    This service implements the IOrchestrationService interface but provides
    minimal orchestration functionality, suitable for simple single-agent workflows.
    """

    def __init__(self):
        """Initialize null orchestration service."""
        logger.info("Initialized NullOrchestrationService - complex orchestration disabled")

    def run_graph(self, graph: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a graph.

        Args:
        ----
            graph: Orchestration graph to run
            inputs: Input data for the graph

        Returns:
        -------
            Result indicating the graph was skipped

        """
        logger.debug("NullOrchestrationService.run_graph called - skipping complex orchestration")

        return {
            "graph_id": f"null_graph_{id(graph)}",
            "status": "skipped",
            "outputs": inputs,  # Pass through inputs as outputs
            "message": "Complex orchestration disabled - using simple pass-through",
            "nodes_executed": 0,
            "execution_time": 0.0,
        }

    def create_agent_node(self, agent_config: Any) -> Any:
        """
        Create an agent node.

        Args:
        ----
            agent_config: Configuration for the agent

        Returns:
        -------
            None (no node created)

        """
        logger.debug("NullOrchestrationService.create_agent_node called - no node created")
        return None

    def is_available(self) -> bool:
        """
        Check if orchestration service is available.

        Returns
        -------
            False (null service provides minimal functionality)

        """
        return False


# Registry of all fallback services
FALLBACK_SERVICES = {
    "IGASAService": NullGASAService,
    "IMonitoringService": NullMonitoringService,
    "ISelfHealingService": NullSelfHealingService,
    "IOrchestrationService": NullOrchestrationService,
}


def get_fallback_service(interface: str) -> Any:
    """
    Get a fallback service for the given interface.

    Args:
    ----
        interface: String name of the service interface

    Returns:
    -------
        Fallback service instance or None if no fallback available

    """
    fallback_class = FALLBACK_SERVICES.get(interface)
    if fallback_class:
        return fallback_class()
    else:
        logger.warning(f"No fallback service available for {interface}")
        return None


__all__ = [
    "NullGASAService",
    "NullMonitoringService",
    "NullSelfHealingService",
    "NullOrchestrationService",
    "FALLBACK_SERVICES",
    "get_fallback_service",
]
