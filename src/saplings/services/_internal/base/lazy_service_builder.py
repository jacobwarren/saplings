from __future__ import annotations

"""
Lazy Service Builder module for Saplings.

This module provides a builder class for creating LazyService instances with
proper configuration and dependency injection. It extends the ServiceBuilder
class to add support for lazy initialization.
"""

import logging
from typing import Any, Generic, Type, TypeVar

from saplings.core._internal.builder import ServiceBuilder
from saplings.core._internal.exceptions import InitializationError
from saplings.services._internal.base.lazy_service import LazyService
from saplings.services._internal.base.service_dependency_graph import ServiceDependencyGraph

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=LazyService)


class LazyServiceBuilder(ServiceBuilder[T], Generic[T]):
    """
    Builder for LazyService instances.

    This class extends the ServiceBuilder class to add support for lazy initialization
    and dependency tracking. It provides methods for registering dependencies and
    building service instances with proper initialization.

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
                    .with_lazy_initialization(True) \
                    .build()
    ```

    """

    def __init__(self, service_class: Type[T]) -> None:
        """
        Initialize the lazy service builder.

        Args:
        ----
            service_class: The class of the service to build

        """
        super().__init__(service_class)
        self._lazy_initialization = True
        self._dependency_graph = ServiceDependencyGraph()
        self._service_name = service_class.__name__
        self._dependency_graph.add_service(self._service_name)

    def with_lazy_initialization(self, lazy: bool) -> LazyServiceBuilder[T]:
        """
        Set whether to use lazy initialization.

        Args:
        ----
            lazy: Whether to use lazy initialization

        Returns:
        -------
            The builder instance for method chaining

        """
        self._lazy_initialization = lazy
        return self

    def with_dependency(self, name: str, dependency: Any) -> LazyServiceBuilder[T]:
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
        # Register the dependency with the parent class
        super().with_dependency(name, dependency)

        # If the dependency is a LazyService, add it to the dependency graph
        if isinstance(dependency, LazyService):
            dependency_name = dependency.__class__.__name__
            self._dependency_graph.add_service(dependency_name)
            try:
                self._dependency_graph.add_dependency(self._service_name, dependency_name)
            except Exception as e:
                logger.warning(
                    f"Could not add dependency {dependency_name} for {self._service_name}: {e}"
                )

        return self

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
            # Create the service instance
            instance = super().build()

            # Register dependencies with the service
            for name, dependency in self._dependencies.items():
                if isinstance(instance, LazyService):
                    instance.register_dependency(name, dependency)

            return instance
        except Exception as e:
            if isinstance(e, InitializationError):
                raise
            raise InitializationError(
                f"Failed to initialize {self._service_class.__name__}: {e}",
                cause=e,
            )
