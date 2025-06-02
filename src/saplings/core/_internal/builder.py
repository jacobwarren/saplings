from __future__ import annotations

"""
Builder module for Saplings services.

This module provides a base ServiceBuilder class that can be extended for specific services
to simplify initialization and configuration. The builder pattern helps separate
configuration from initialization and provides a fluent interface for service creation.
"""

import logging
from typing import Any, Dict, Generic, Type, TypeVar

from saplings.core._internal.exceptions import InitializationError
from saplings.core._internal.validation.validation import validate_required

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceBuilder(Generic[T]):
    """
    Base builder class for Saplings services.

    This class provides a fluent interface for building service instances with
    proper configuration and dependency injection. It separates configuration
    from initialization and ensures all required dependencies are provided.

    Example:
    -------
    ```python
    # Create a builder for ExecutionService
    builder = ExecutionServiceBuilder()

    # Configure the builder with dependencies and options
    service = builder.with_model(model) \
                    .with_gasa(gasa_service) \
                    .with_validator(validator_service) \
                    .with_config(config) \
                    .build()
    ```

    """

    def __init__(self, service_class: Type[T]) -> None:
        """
        Initialize the service builder.

        Args:
        ----
            service_class: The class of the service to build

        """
        self._service_class = service_class
        self._dependencies: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
        self._required_dependencies: list[str] = []

    def with_dependency(self, name: str, dependency: Any) -> ServiceBuilder[T]:
        """
        Add a dependency to the service.

        Args:
        ----
            name: Name of the dependency
            dependency: The dependency instance

        Returns:
        -------
            The builder instance for method chaining

        """
        self._dependencies[name] = dependency
        return self

    def with_config(self, config: Dict[str, Any]) -> ServiceBuilder[T]:
        """
        Add configuration to the service.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            The builder instance for method chaining

        """
        self._config.update(config)
        return self

    def require_dependency(self, name: str) -> None:
        """
        Mark a dependency as required.

        Args:
        ----
            name: Name of the required dependency

        """
        if name not in self._required_dependencies:
            self._required_dependencies.append(name)

    def _validate_dependencies(self) -> None:
        """
        Validate that all required dependencies are provided.

        Raises
        ------
            InitializationError: If a required dependency is missing

        """
        for name in self._required_dependencies:
            try:
                validate_required(self._dependencies.get(name), f"dependency '{name}'")
            except ValueError as e:
                raise InitializationError(
                    f"Missing required dependency '{name}' for {self._service_class.__name__}",
                    cause=e,
                )

    def build(self) -> T:
        """
        Build the service instance with the configured dependencies.

        Returns
        -------
            The initialized service instance

        Raises
        ------
            InitializationError: If service initialization fails

        """
        try:
            # Validate required dependencies
            self._validate_dependencies()

            # Create kwargs by combining dependencies and config
            kwargs = {**self._dependencies, **self._config}

            # Create the service instance
            instance = self._service_class(**kwargs)
            logger.debug(f"Built {self._service_class.__name__} with {len(kwargs)} parameters")
            return instance
        except Exception as e:
            if isinstance(e, InitializationError):
                raise
            raise InitializationError(
                f"Failed to initialize {self._service_class.__name__}: {e}",
                cause=e,
            )
