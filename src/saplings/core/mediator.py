from __future__ import annotations

"""
Mediator pattern implementation for Saplings.

This module provides a mediator pattern implementation for service communication.
"""

from enum import Enum

from saplings.core._internal.mediator import (
    ServiceMediator,
    ServiceRequest,
    ServiceResponse,
    get_service_mediator,
)


class ServiceRequestType(str, Enum):
    """Types of service requests."""

    VALIDATION = "validation"
    EXECUTION = "execution"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    JUDGE = "judge"
    MONITORING = "monitoring"
    SELF_HEALING = "self_healing"
    TOOL = "tool"
    MODEL = "model"
    CUSTOM = "custom"


__all__ = [
    "ServiceMediator",
    "ServiceRequest",
    "ServiceRequestType",
    "ServiceResponse",
    "get_service_mediator",
]
