from __future__ import annotations

"""
Base Service API module for Saplings.

This module provides the base service class.
"""

from saplings.api.stability import stable


@stable
class Service:
    """
    Base class for all services.

    This class provides common functionality for all services.
    """

    def __init__(self, **kwargs):
        """
        Initialize the service.

        Args:
        ----
            **kwargs: Additional configuration options

        """
        self.config = kwargs
        self._name = kwargs.get("name", self.__class__.__name__)

    @property
    def name(self) -> str:
        """
        Get the name of the service.

        Returns
        -------
            str: Name of the service

        """
        return self._name


__all__ = [
    "Service",
]
