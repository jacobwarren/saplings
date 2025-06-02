"""
Service availability detection and conditional registration.

This module provides functionality to detect which optional services are available
and register them conditionally based on dependency availability.

This implementation follows the patterns established in Task 3.4 for graceful
degradation when optional services are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ServiceAvailability:
    """
    Manages service availability detection and conditional registration.

    This class provides a centralized way to register services only when their
    dependencies are available, and fall back to null implementations when they're not.
    """

    def __init__(self):
        """Initialize service availability manager."""
        self.available_services: Dict[str, Any] = {}
        self.fallback_services: Dict[str, Any] = {}

    def register_conditional_service(
        self, interface: str, implementation: Any, condition: Callable[[], bool]
    ) -> None:
        """
        Register service only if condition is met.

        Args:
        ----
            interface: String name of the service interface
            implementation: Service implementation to register
            condition: Function that returns True if service should be registered

        """
        if condition():
            self.available_services[interface] = implementation
            logger.debug(f"Registered {interface} with real implementation")
        else:
            # Use fallback if available
            fallback = self.fallback_services.get(interface)
            self.available_services[interface] = fallback
            if fallback:
                logger.warning(f"Using fallback implementation for {interface}")
            else:
                logger.warning(f"No fallback available for {interface}")

    def register_fallback(self, interface: str, fallback_implementation: Any) -> None:
        """
        Register a fallback implementation for a service interface.

        Args:
        ----
            interface: String name of the service interface
            fallback_implementation: Fallback service implementation

        """
        self.fallback_services[interface] = fallback_implementation
        logger.debug(f"Registered fallback for {interface}")

    def get_service(self, interface: str) -> Optional[Any]:
        """
        Get the available service for an interface.

        Args:
        ----
            interface: String name of the service interface

        Returns:
        -------
            Service implementation or None if not available

        """
        return self.available_services.get(interface)

    def is_service_available(self, interface: str) -> bool:
        """
        Check if a service is available (not using fallback).

        Args:
        ----
            interface: String name of the service interface

        Returns:
        -------
            True if real implementation is available, False if using fallback or None

        """
        service = self.available_services.get(interface)
        fallback = self.fallback_services.get(interface)

        # Service is "available" if it exists and is not the fallback
        return service is not None and service != fallback


# Global service availability manager
_service_availability = ServiceAvailability()


def register_conditional_service(
    interface: str, implementation: Any, condition: Callable[[], bool]
) -> None:
    """
    Register service only if condition is met (global function).

    Args:
    ----
        interface: String name of the service interface
        implementation: Service implementation to register
        condition: Function that returns True if service should be registered

    """
    _service_availability.register_conditional_service(interface, implementation, condition)


def register_fallback_service(interface: str, fallback_implementation: Any) -> None:
    """
    Register a fallback implementation for a service interface (global function).

    Args:
    ----
        interface: String name of the service interface
        fallback_implementation: Fallback service implementation

    """
    _service_availability.register_fallback(interface, fallback_implementation)


def get_available_services() -> Dict[str, Any]:
    """
    Get all available services.

    Returns
    -------
        Dictionary mapping interface names to service implementations

    """
    return _service_availability.available_services.copy()


def get_service(interface: str) -> Optional[Any]:
    """
    Get the available service for an interface (global function).

    Args:
    ----
        interface: String name of the service interface

    Returns:
    -------
        Service implementation or None if not available

    """
    return _service_availability.get_service(interface)


def is_service_available(interface: str) -> bool:
    """
    Check if a service is available (not using fallback) (global function).

    Args:
    ----
        interface: String name of the service interface

    Returns:
    -------
        True if real implementation is available, False if using fallback or None

    """
    return _service_availability.is_service_available(interface)


def check_optional_service_availability() -> Dict[str, bool]:
    """
    Check availability of all optional services.

    Returns
    -------
        Dictionary mapping service names to availability status

    """
    try:
        from saplings._internal.optional_deps import check_feature_availability

        features = check_feature_availability()

        # Map features to service interfaces
        service_availability = {
            "IGASAService": features.get("gasa", False),
            "IMonitoringService": features.get("monitoring", False),
            "ISelfHealingService": True,  # Self-healing doesn't require external deps
            "IOrchestrationService": True,  # Orchestration is core functionality
            "IModalityService": True,  # Modality service has fallbacks
        }

        return service_availability

    except ImportError:
        logger.warning("Optional dependencies module not available")
        return {}


__all__ = [
    "ServiceAvailability",
    "register_conditional_service",
    "register_fallback_service",
    "get_available_services",
    "get_service",
    "is_service_available",
    "check_optional_service_availability",
]
