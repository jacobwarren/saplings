from __future__ import annotations

"""
Monitoring types for Saplings.

This module provides type definitions for monitoring services.
"""

from enum import Enum, auto


class TracingBackend(Enum):
    """
    Tracing backend options.

    This enum defines the available tracing backends for monitoring services.
    """

    CONSOLE = auto()
    """Console tracing backend."""

    FILE = auto()
    """File tracing backend."""

    LANGSMITH = auto()
    """LangSmith tracing backend."""

    CUSTOM = auto()
    """Custom tracing backend."""
