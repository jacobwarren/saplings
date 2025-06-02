from __future__ import annotations

"""
Common locator module for Saplings registry.

This module provides common functionality for service location
to avoid circular dependencies between registry and service modules.
"""

from typing import Any, Dict


class ServiceLocator:
    """Service locator for Saplings."""

    _instance = None
    _services: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "ServiceLocator":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_service(self, name: str, service: Any) -> None:
        """Register a service."""
        self._services[name] = service

    def get_service(self, name: str) -> Any:
        """Get a service by name."""
        return self._services.get(name)


def get_service_locator() -> ServiceLocator:
    """
    Get the service locator singleton instance.

    Returns
    -------
        ServiceLocator: The service locator instance

    """
    return ServiceLocator.get_instance()
