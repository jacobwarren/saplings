from __future__ import annotations

"""
Base service builder for Saplings.

This module provides a base builder for services in the Saplings framework.
"""

import logging
from typing import Any, Dict, Generic, Type, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ServiceBuilder(Generic[T]):
    """
    Base builder for services.

    This class provides a base builder for services in the Saplings framework.
    It implements the builder pattern for service initialization.
    """

    def __init__(self, service_type: Type[T]) -> None:
        """
        Initialize the service builder.

        Args:
        ----
            service_type: The type of service to build

        """
        self._service_type = service_type
        self._dependencies: Dict[str, Any] = {}

    def with_dependency(self, name: str, value: Any) -> ServiceBuilder[T]:
        """
        Add a dependency to the service.

        Args:
        ----
            name: Name of the dependency
            value: Value of the dependency

        Returns:
        -------
            The builder instance for method chaining

        """
        self._dependencies[name] = value
        return self

    def build(self) -> T:
        """
        Build the service instance with the configured dependencies.

        Returns
        -------
            The initialized service instance

        """
        logger.debug(f"Building service of type {self._service_type.__name__}")

        try:
            # Create the service instance with the dependencies
            service = self._service_type(**self._dependencies)
            logger.debug(f"Service {self._service_type.__name__} built successfully")
            return service
        except Exception as e:
            logger.error(f"Error building service {self._service_type.__name__}: {e}")
            raise
