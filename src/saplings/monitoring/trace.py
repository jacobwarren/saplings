"""
Trace module for Saplings monitoring.

This module provides the core tracing infrastructure for monitoring.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from saplings.monitoring.config import MonitoringConfig, TracingBackend

logger = logging.getLogger(__name__)

try:
    import opentelemetry.trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning(
        "OpenTelemetry not installed. OTEL tracing will not be available. "
        "Install OpenTelemetry with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
    )


class SpanEvent:
    """Event within a span."""

    def __init__(
        self,
        name: str,
        timestamp: Optional[datetime] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a span event.

        Args:
            name: Name of the event
            timestamp: Timestamp of the event
            attributes: Attributes of the event
        """
        self.name = name
        self.timestamp = timestamp or datetime.now()
        self.attributes = attributes or {}


class SpanContext:
    """Context for a span."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_id: Optional[str] = None,
    ):
        """
        Initialize a span context.

        Args:
            trace_id: ID of the trace
            span_id: ID of the span
            parent_id: ID of the parent span
        """
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id


class Span:
    """Span representing a unit of work."""

    def __init__(
        self,
        name: str,
        context: SpanContext,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a span.

        Args:
            name: Name of the span
            context: Context of the span
            start_time: Start time of the span
            end_time: End time of the span
            status: Status of the span
            attributes: Attributes of the span
        """
        self.name = name
        self.context = context
        self.span_id = context.span_id
        self.parent_id = context.parent_id
        self.trace_id = context.trace_id
        self.start_time = start_time or datetime.now()
        self.end_time = end_time
        self.status = status
        self.attributes = attributes or {}
        self.events: List[SpanEvent] = []

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SpanEvent:
        """
        Add an event to the span.

        Args:
            name: Name of the event
            attributes: Attributes of the event

        Returns:
            SpanEvent: The created event
        """
        event = SpanEvent(name=name, attributes=attributes)
        self.events.append(event)

        # Add event to OTEL span if available
        if hasattr(self, "otel_span"):
            self.otel_span.add_event(
                name=name,
                attributes=attributes,
                timestamp=event.timestamp,
            )

        return event

    def set_status(self, status: str) -> None:
        """
        Set the status of the span.

        Args:
            status: Status of the span
        """
        self.status = status

        # Set status on OTEL span if available
        if hasattr(self, "otel_span"):
            if status == "ERROR":
                self.otel_span.set_status(
                    otel_trace.Status(
                        status_code=otel_trace.StatusCode.ERROR,
                        description=self.attributes.get("error.message", "An error occurred"),
                    )
                )
            elif status == "OK":
                self.otel_span.set_status(
                    otel_trace.Status(status_code=otel_trace.StatusCode.OK)
                )

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set an attribute of the span.

        Args:
            key: Key of the attribute
            value: Value of the attribute
        """
        self.attributes[key] = value

        # Set attribute on OTEL span if available
        if hasattr(self, "otel_span"):
            self.otel_span.set_attribute(key, value)

    def end(self) -> None:
        """End the span."""
        if not self.end_time:
            self.end_time = datetime.now()

    def duration_ms(self) -> float:
        """
        Get the duration of the span in milliseconds.

        Returns:
            float: Duration in milliseconds
        """
        if not self.end_time:
            return 0.0

        return (self.end_time - self.start_time).total_seconds() * 1000.0


class Trace:
    """Trace representing a collection of spans."""

    def __init__(
        self,
        trace_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        status: str = "OK",
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a trace.

        Args:
            trace_id: ID of the trace
            start_time: Start time of the trace
            end_time: End time of the trace
            status: Status of the trace
            attributes: Attributes of the trace
        """
        self.trace_id = trace_id
        self.start_time = start_time or datetime.now()
        self.end_time = end_time
        self.status = status
        self.attributes = attributes or {}
        self.spans: List[Span] = []

    def add_span(self, span: Span) -> None:
        """
        Add a span to the trace.

        Args:
            span: Span to add
        """
        self.spans.append(span)

        # Update trace status if span has error
        if span.status == "ERROR" and self.status != "ERROR":
            self.status = "ERROR"

    def end(self) -> None:
        """End the trace."""
        if not self.end_time:
            self.end_time = datetime.now()

    def duration_ms(self) -> float:
        """
        Get the duration of the trace in milliseconds.

        Returns:
            float: Duration in milliseconds
        """
        if not self.end_time:
            return 0.0

        return (self.end_time - self.start_time).total_seconds() * 1000.0

    def get_root_spans(self) -> List[Span]:
        """
        Get the root spans of the trace.

        Returns:
            List[Span]: Root spans
        """
        return [span for span in self.spans if not span.parent_id]

    def get_child_spans(self, parent_id: str) -> List[Span]:
        """
        Get the child spans of a parent span.

        Args:
            parent_id: ID of the parent span

        Returns:
            List[Span]: Child spans
        """
        return [span for span in self.spans if span.parent_id == parent_id]


class TraceManager:
    """
    Manager for traces.

    This class manages the lifecycle of traces and spans, providing methods
    for creating, updating, and querying trace data. It also supports callbacks
    for trace lifecycle events, enabling integration with monitoring systems
    like LangSmith.
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
    ):
        """
        Initialize the trace manager.

        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.traces: Dict[str, Trace] = {}
        self.active_traces: Set[str] = set()
        self.active_spans: Dict[str, Span] = {}

        # Callbacks for trace lifecycle events
        self.trace_callbacks: List[Callable[[str, str], None]] = []

        # Initialize OpenTelemetry if available and enabled
        self.otel_tracer = None
        if self.config.tracing_backend == TracingBackend.OTEL:
            self._init_otel()

    def _init_otel(self) -> None:
        """Initialize OpenTelemetry."""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not installed. OTEL tracing will not be available.")
            return

        try:
            # Create a tracer provider
            provider = TracerProvider()

            # Add console exporter
            if self.config.tracing_backend == TracingBackend.CONSOLE:
                console_exporter = ConsoleSpanExporter()
                processor = BatchSpanProcessor(console_exporter)
                provider.add_span_processor(processor)

            # Add OTLP exporter if endpoint is configured
            if self.config.tracing_backend == TracingBackend.OTEL and self.config.otel_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.config.otel_endpoint)
                processor = BatchSpanProcessor(otlp_exporter)
                provider.add_span_processor(processor)

            # Set the tracer provider
            otel_trace.set_tracer_provider(provider)

            # Create a tracer
            self.otel_tracer = otel_trace.get_tracer("saplings")

            logger.info("OpenTelemetry initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.otel_tracer = None

    def register_trace_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """
        Register a callback for trace lifecycle events.

        The callback will be called with the trace ID and event type
        when a trace is created, completed, or encounters an error.

        Args:
            callback: Function to call with (trace_id, event_type)
        """
        if callback not in self.trace_callbacks:
            self.trace_callbacks.append(callback)
            logger.debug(f"Registered trace callback: {callback.__name__}")

    def unregister_trace_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> bool:
        """
        Unregister a trace callback.

        Args:
            callback: Function to unregister

        Returns:
            bool: True if the callback was removed, False if not found
        """
        if callback in self.trace_callbacks:
            self.trace_callbacks.remove(callback)
            logger.debug(f"Unregistered trace callback: {callback.__name__}")
            return True
        return False

    def _trigger_callbacks(
        self,
        trace_id: str,
        event: str,
    ) -> None:
        """
        Trigger all registered callbacks for a trace event.

        Args:
            trace_id: ID of the trace
            event: Event type (e.g., "created", "completed", "error")
        """
        for callback in self.trace_callbacks:
            try:
                callback(trace_id, event)
            except Exception as e:
                logger.error(f"Error in trace callback {callback.__name__}: {e}")

    def create_trace(
        self,
        trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Trace:
        """
        Create a new trace.

        Args:
            trace_id: ID of the trace (generated if not provided)
            attributes: Attributes of the trace

        Returns:
            Trace: The created trace
        """
        trace_id = trace_id or str(uuid.uuid4())
        trace = Trace(trace_id=trace_id, attributes=attributes)
        self.traces[trace_id] = trace
        self.active_traces.add(trace_id)

        # Trigger callbacks
        self._trigger_callbacks(trace_id, "created")

        return trace

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """
        Start a new span.

        Args:
            name: Name of the span
            trace_id: ID of the trace (created if not provided)
            parent_id: ID of the parent span
            attributes: Attributes of the span

        Returns:
            Span: The created span
        """
        # Create trace if not provided
        if not trace_id:
            trace = self.create_trace()
            trace_id = trace.trace_id
        elif trace_id not in self.traces:
            trace = self.create_trace(trace_id=trace_id)
        else:
            trace = self.traces[trace_id]

        # Create span
        span_id = str(uuid.uuid4())
        context = SpanContext(trace_id=trace_id, span_id=span_id, parent_id=parent_id)
        span = Span(name=name, context=context, attributes=attributes)

        # Add span to trace
        trace.add_span(span)

        # Add span to active spans
        self.active_spans[span_id] = span

        # Create OTEL span if enabled
        if self.otel_tracer:
            # Create OTEL context
            otel_context = None

            # If parent_id is provided, create a parent context
            if parent_id:
                # Get the parent span
                parent_span = self.get_span(parent_id)
                if parent_span and hasattr(parent_span, "otel_span"):
                    # Use the parent's OTEL context
                    otel_context = parent_span.otel_span.get_span_context()

            # Determine span kind
            span_kind = otel_trace.SpanKind.INTERNAL
            if "kind" in attributes:
                kind = attributes.get("kind", "internal").lower()
                if kind == "server":
                    span_kind = otel_trace.SpanKind.SERVER
                elif kind == "client":
                    span_kind = otel_trace.SpanKind.CLIENT
                elif kind == "producer":
                    span_kind = otel_trace.SpanKind.PRODUCER
                elif kind == "consumer":
                    span_kind = otel_trace.SpanKind.CONSUMER

            # Start the OTEL span
            otel_span = self.otel_tracer.start_span(
                name=name,
                context=otel_context,
                kind=span_kind,
                attributes=attributes,
                start_time=span.start_time,
            )

            # Store the OTEL span in our span object for later reference
            span.otel_span = otel_span

        return span

    def end_span(
        self,
        span_id: str,
        status: Optional[str] = None,
    ) -> None:
        """
        End a span.

        Args:
            span_id: ID of the span to end
            status: Final status of the span
        """
        if span_id not in self.active_spans:
            logger.warning(f"Span {span_id} not found in active spans")
            return

        span = self.active_spans[span_id]

        # Set status if provided
        if status:
            span.set_status(status)

            # Set OTEL span status if available
            if hasattr(span, "otel_span") and self.otel_tracer:
                if status == "ERROR":
                    span.otel_span.set_status(
                        otel_trace.Status(
                            status_code=otel_trace.StatusCode.ERROR,
                            description=span.attributes.get("error.message", "An error occurred"),
                        )
                    )
                elif status == "OK":
                    span.otel_span.set_status(
                        otel_trace.Status(status_code=otel_trace.StatusCode.OK)
                    )

        # End span
        span.end()

        # End OTEL span if available
        if hasattr(span, "otel_span") and self.otel_tracer:
            # Set any final attributes
            for key, value in span.attributes.items():
                span.otel_span.set_attribute(key, value)

            # Record events
            for event in span.events:
                span.otel_span.add_event(
                    name=event.name,
                    attributes=event.attributes,
                    timestamp=event.timestamp,
                )

            # End the span
            span.otel_span.end(end_time=span.end_time)

        # Remove from active spans
        del self.active_spans[span_id]

        # End trace if all spans are done
        trace = self.traces[span.trace_id]

        # Update trace status if span has error
        if status == "ERROR":
            trace.status = "ERROR"

        if not any(s.span_id in self.active_spans for s in trace.spans):
            trace.end()
            if trace.trace_id in self.active_traces:
                self.active_traces.remove(trace.trace_id)

                # Trigger callbacks for trace completion
                event = "error" if trace.status == "ERROR" else "completed"
                self._trigger_callbacks(trace.trace_id, event)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """
        Get a trace by ID.

        Args:
            trace_id: ID of the trace

        Returns:
            Optional[Trace]: The trace if found
        """
        return self.traces.get(trace_id)

    def get_span(self, span_id: str) -> Optional[Span]:
        """
        Get a span by ID.

        Args:
            span_id: ID of the span

        Returns:
            Optional[Span]: The span if found
        """
        # Check active spans first
        if span_id in self.active_spans:
            return self.active_spans[span_id]

        # Search in all traces
        for trace in self.traces.values():
            for span in trace.spans:
                if span.span_id == span_id:
                    return span

        return None

    def list_traces(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Trace]:
        """
        List all traces within a time range.

        Args:
            start_time: Start time for filtering
            end_time: End time for filtering

        Returns:
            List[Trace]: Matching traces
        """
        traces = list(self.traces.values())

        # Filter by start time
        if start_time:
            traces = [t for t in traces if t.start_time >= start_time]

        # Filter by end time
        if end_time:
            traces = [t for t in traces if t.end_time and t.end_time <= end_time]

        return traces

    def clear_traces(
        self,
        before_time: Optional[datetime] = None,
    ) -> int:
        """
        Clear traces from memory.

        Args:
            before_time: Clear traces that ended before this time

        Returns:
            int: Number of traces cleared
        """
        if not before_time:
            # Clear all inactive traces
            inactive_traces = [
                trace_id for trace_id in self.traces
                if trace_id not in self.active_traces
            ]

            for trace_id in inactive_traces:
                del self.traces[trace_id]

            return len(inactive_traces)
        else:
            # Clear traces that ended before the specified time
            traces_to_clear = [
                trace_id for trace_id, trace in self.traces.items()
                if trace.end_time and trace.end_time < before_time
                and trace_id not in self.active_traces
            ]

            for trace_id in traces_to_clear:
                del self.traces[trace_id]

            return len(traces_to_clear)

    def export_traces(
        self,
        output_path: str,
        trace_ids: Optional[List[str]] = None,
    ) -> int:
        """
        Export traces to a file.

        Args:
            output_path: Path to save the traces
            trace_ids: IDs of traces to export (all if not provided)

        Returns:
            int: Number of traces exported
        """
        # Determine which traces to export
        if trace_ids:
            traces_to_export = [
                trace for trace_id, trace in self.traces.items()
                if trace_id in trace_ids
            ]
        else:
            traces_to_export = list(self.traces.values())

        # Convert traces to dictionaries
        trace_dicts = []
        for trace in traces_to_export:
            # Convert spans to dictionaries
            span_dicts = []
            for span in trace.spans:
                # Convert events to dictionaries
                event_dicts = []
                for event in span.events:
                    event_dicts.append({
                        "name": event.name,
                        "timestamp": event.timestamp.isoformat(),
                        "attributes": event.attributes,
                    })

                span_dicts.append({
                    "span_id": span.span_id,
                    "name": span.name,
                    "trace_id": span.trace_id,
                    "parent_id": span.parent_id,
                    "start_time": span.start_time.isoformat(),
                    "end_time": span.end_time.isoformat() if span.end_time else None,
                    "status": span.status,
                    "attributes": span.attributes,
                    "events": event_dicts,
                })

            trace_dicts.append({
                "trace_id": trace.trace_id,
                "start_time": trace.start_time.isoformat(),
                "end_time": trace.end_time.isoformat() if trace.end_time else None,
                "status": trace.status,
                "attributes": trace.attributes,
                "spans": span_dicts,
            })

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to file
        with open(output_path, "w") as f:
            json.dump(trace_dicts, f, indent=2)

        logger.info(f"Exported {len(trace_dicts)} traces to {output_path}")

        return len(trace_dicts)
