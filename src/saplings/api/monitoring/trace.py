from __future__ import annotations

"""
Trace API module for Saplings.

This module provides the public API for trace components.
"""

from saplings.api.stability import beta
from saplings.monitoring._internal.trace import Span as _Span
from saplings.monitoring._internal.trace import SpanContext as _SpanContext
from saplings.monitoring._internal.trace import Trace as _Trace
from saplings.monitoring._internal.trace import TraceManager as _TraceManager


@beta
class SpanContext(_SpanContext):
    """
    Context for a span.

    A span context contains the trace ID, span ID, and parent ID for a span.
    It is used to link spans together in a trace.
    """


@beta
class Span(_Span):
    """
    Span representing a unit of work.

    A span represents a single operation or unit of work in a trace. It has
    a name, start and end times, status, and attributes. Spans can be nested
    to represent parent-child relationships between operations.
    """


@beta
class Trace(_Trace):
    """
    Trace representing a collection of spans.

    A trace represents a complete request or operation, consisting of multiple
    spans. It has a trace ID, start and end times, status, and attributes.
    """


@beta
class TraceManager(_TraceManager):
    """
    Manager for traces.

    This class manages the lifecycle of traces and spans, providing methods
    for creating, updating, and querying trace data. It also supports callbacks
    for trace lifecycle events, enabling integration with monitoring systems
    like LangSmith.
    """


__all__ = [
    "SpanContext",
    "Span",
    "Trace",
    "TraceManager",
]
