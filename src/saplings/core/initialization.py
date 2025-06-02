from __future__ import annotations

"""
Initialization utilities for Saplings.

This module provides utilities for initializing and managing services using
an event-based initialization system. This system automatically handles
dependencies between services and ensures they are initialized in the correct
order.
"""

from saplings.core._internal.initialization import (
    dispose_service,
    get_initialization_order,
    get_service_state,
    initialize_service,
    mark_service_ready,
    register_dependency,
    shutdown_service,
)
from saplings.core._internal.service_registry import get_service_registry
from saplings.core.lifecycle import ServiceLifecycle, ServiceState


def get_all_services():
    """
    Get all registered services and their states.

    Returns
    -------
        Dict[str, ServiceState]: Dictionary of service names and states

    """
    registry = get_service_registry()
    return registry.get_all_services()


__all__ = [
    "ServiceLifecycle",
    "ServiceState",
    "dispose_service",
    "get_all_services",
    "get_initialization_order",
    "get_service_state",
    "initialize_service",
    "mark_service_ready",
    "register_dependency",
    "shutdown_service",
]
