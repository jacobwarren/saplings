from __future__ import annotations

"""
Service lifecycle management for Saplings.

This module provides classes and utilities for managing the lifecycle of services,
including state tracking, validation, and cleanup.
"""

from saplings.core._internal.lifecycle import (
    ServiceLifecycle,
    ServiceState,
    validate_service_state,
)

__all__ = [
    "ServiceLifecycle",
    "ServiceState",
    "validate_service_state",
]
