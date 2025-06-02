from __future__ import annotations

"""
Monitoring service API module for Saplings.

This module provides the public API for monitoring services.
"""

from saplings.api.core.interfaces import IMonitoringService as _IMonitoringService
from saplings.api.stability import beta, stable
from saplings.monitoring._internal.service import MonitoringService as _MonitoringService


@stable
class IMonitoringService(_IMonitoringService):
    """
    Interface for monitoring services.

    This interface defines the contract for monitoring services, which
    provide functionality for monitoring and tracing operations.
    """


@beta
class MonitoringService(_MonitoringService):
    """
    Service for monitoring.

    This class provides functionality for monitoring and tracing operations,
    including creating traces, recording spans, and generating visualizations.
    """


__all__ = [
    "IMonitoringService",
    "MonitoringService",
]
