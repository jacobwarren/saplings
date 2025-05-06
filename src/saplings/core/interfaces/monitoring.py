from __future__ import annotations

"""
Monitoring service interface for Saplings.

This module defines the interface for monitoring operations that track
execution metrics and traces. This is a pure interface with no implementation
details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any


class IMonitoringService(ABC):
    """Interface for monitoring operations."""

    @property
    @abstractmethod
    def enabled(self):
        """
        Whether monitoring is enabled.

        Returns
        -------
            bool: Enabled status

        """

    @property
    @abstractmethod
    def trace_manager(self):
        """
        Get the trace manager if enabled.

        Returns
        -------
            Any: Trace manager instance

        """

    @abstractmethod
    def create_trace(self):
        """
        Create a new trace.

        Returns
        -------
            Dict[str, Any]: Created trace

        """

    @abstractmethod
    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """
        Start a new span in the trace.

        Args:
        ----
            name: Span name
            trace_id: Optional trace ID
            parent_id: Optional parent span ID
            attributes: Optional span attributes

        Returns:
        -------
            Any: The created span

        """

    @abstractmethod
    def end_span(self, span_id: str) -> None:
        """
        End a span in the trace.

        Args:
        ----
            span_id: ID of the span to end

        """

    @abstractmethod
    def process_trace(self, trace_id: str) -> Any:
        """
        Process a trace for analysis.

        Args:
        ----
            trace_id: ID of the trace to process

        Returns:
        -------
            Any: Processed trace

        """

    @abstractmethod
    def identify_bottlenecks(
        self, threshold_ms: float = 100.0, min_call_count: int = 1
    ) -> list[dict[str, Any]]:
        """
        Identify performance bottlenecks.

        Args:
        ----
            threshold_ms: Minimum duration in milliseconds to consider a bottleneck
            min_call_count: Minimum number of calls to consider

        Returns:
        -------
            List[Dict[str, Any]]: Identified bottlenecks

        """

    @abstractmethod
    def identify_error_sources(
        self, min_error_rate: float = 0.1, min_call_count: int = 1
    ) -> list[dict[str, Any]]:
        """
        Identify error sources.

        Args:
        ----
            min_error_rate: Minimum error rate to consider
            min_call_count: Minimum number of calls to consider

        Returns:
        -------
            List[Dict[str, Any]]: Identified error sources

        """

    @abstractmethod
    def log_event(
        self,
        event_type: str,
        data: dict[str, Any],
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        """
        Log an event.

        Args:
        ----
            event_type: Type of event
            data: Event data
            trace_id: Optional trace ID
            span_id: Optional span ID

        """
