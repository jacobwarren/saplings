from __future__ import annotations

"""
Core event system for Saplings.

This module provides a centralized event system for cross-service communication,
helping to reduce direct dependencies between services and enabling a more
loosely coupled architecture.
"""

from saplings.core._internal.events import (
    AsyncEventHandler,
    CoreEvent,
    CoreEventBus,
    CoreEventType,
    EventHandler,
    get_event_bus,
)

__all__ = [
    "AsyncEventHandler",
    "CoreEvent",
    "CoreEventBus",
    "CoreEventType",
    "EventHandler",
    "get_event_bus",
]
