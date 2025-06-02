from __future__ import annotations

"""
Monitoring service interface for Saplings.

This module defines the interface for monitoring operations that track
execution metrics and traces. This is a pure interface with no implementation
details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MonitoringConfig:
    """Configuration for monitoring operations."""

    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    exporter: Optional[str] = None
    sampling_rate: float = 1.0


@dataclass
class MonitoringEvent:
    """Event for monitoring operations."""

    event_type: str
    timestamp: float
    data: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


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

    @abstractmethod
    async def record_event(
        self,
        event: MonitoringEvent,
        config: Optional[MonitoringConfig] = None,
    ) -> bool:
        """
        Record a monitoring event.

        Args:
        ----
            event: The event to record
            config: Optional monitoring configuration

        Returns:
        -------
            bool: True if the event was recorded successfully

        """

    @abstractmethod
    def register_service(self, service_name: str, service_instance: Any) -> None:
        """
        Register a service with the monitoring system.

        This allows the monitoring service to track and interact with other services
        in the system, enabling features like service health monitoring and
        dependency tracking.

        Args:
        ----
            service_name: Name of the service
            service_instance: Service instance to register

        """

    @abstractmethod
    def register_service_hook(self, service_name: str, hook: Any) -> None:
        """
        Register a hook to be called when a service is registered.

        This allows components to be notified when specific services become
        available, enabling event-based initialization and dependency management.

        Args:
        ----
            service_name: Name of the service to hook
            hook: Callback function to execute when the service is registered

        """

    @abstractmethod
    def get_registered_service(self, service_name: str) -> Any:
        """
        Get a registered service by name.

        Args:
        ----
            service_name: Name of the service to retrieve

        Returns:
        -------
            The registered service instance or None if not found

        """

    @abstractmethod
    def get_registered_services(self) -> dict[str, Any]:
        """
        Get all registered services.

        Returns
        -------
            Dictionary of service names to service instances

        """
