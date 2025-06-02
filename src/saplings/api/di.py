from __future__ import annotations

"""
Dependency Injection API module for Saplings.

This module provides the public API for dependency injection.
"""

from typing import Generic, Optional, TypeVar

from saplings.api.stability import beta, stable
from saplings.di._internal import (
    Container as _Container,
)
from saplings.di._internal import (
    container as _container,
)
from saplings.di._internal import (
    inject as _inject,
)
from saplings.di._internal import (
    register as _register,
)
from saplings.di._internal import (
    reset_container as _reset_container,
)
from saplings.di._internal.container import InitializationStatus as _InitializationStatus
from saplings.di._internal.registration import (
    ConfiguredProvider as _ConfiguredProvider,
)
from saplings.di._internal.registration import (
    FactoryProvider as _FactoryProvider,
)
from saplings.di._internal.registration import (
    InitializableProvider as _InitializableProvider,
)
from saplings.di._internal.registration import (
    LazyProvider as _LazyProvider,
)
from saplings.di._internal.registration import (
    Provider as _Provider,
)
from saplings.di._internal.registration import (
    SingletonProvider as _SingletonProvider,
)
from saplings.di._internal.scope import (
    LifecycleScope as _LifecycleScope,
)
from saplings.di._internal.scope import (
    Scope as _Scope,
)

# Type variable for generic type hints
T = TypeVar("T")


@stable
class Service:
    """
    Base protocol for all services.

    This is a marker class that all services should implement or inherit from.
    It provides a common base type for service registration and resolution.
    """

    @property
    def name(self) -> str:
        """Get the name of the service."""
        return self.__class__.__name__


@stable
class Container(_Container):
    """
    Container for dependency injection.

    This class provides a container for dependency injection in the Saplings framework.
    It supports registration, resolution, and initialization of services.

    Features:
    - Circular dependency detection
    - Scoped lifetimes
    - Automatic constructor injection
    - Factory function support
    - Initialization status tracking
    - Lazy dependency resolution
    """

    @beta
    def get_initialization_status(self, service_type):
        """
        Get the initialization status of a service.

        Args:
        ----
            service_type: The type to check

        Returns:
        -------
            The initialization status of the service

        """
        return super().get_initialization_status(service_type)

    @beta
    def initialize_service(self, service_type):
        """
        Initialize a service.

        This method initializes a service and all its dependencies.
        It tracks initialization status to prevent duplicate initialization
        and detects circular dependencies.

        Args:
        ----
            service_type: The service to initialize

        Raises:
        ------
            CircularDependencyError: If a circular dependency is detected
            InitializationError: If the service fails to initialize

        """
        return super().initialize_service(service_type)

    @beta
    def add_initialization_dependency(self, service_type, dependency_type):
        """
        Add an initialization dependency between services.

        Args:
        ----
            service_type: The service that depends on the dependency
            dependency_type: The dependency

        """
        return super().add_initialization_dependency(service_type, dependency_type)


# Re-export the container instance
container = _container


@stable
def get_container():
    """
    Get the global container instance.

    Returns
    -------
        The global container instance

    """
    return container


@stable
class Inject(Generic[T]):
    """
    Explicit marker for dependencies that should be injected.

    This class is used to explicitly mark dependencies that should be
    injected by the container. It can be used in constructor parameters
    to make dependencies more explicit and to support optional dependencies.

    Example:
    -------
    ```python
    class Service:
        def __init__(self, required: IRequired, optional: Inject[IOptional] = None):
            self.required = required
            self.optional = optional
    ```

    """

    def __init__(self, value: Optional[T] = None):
        """
        Initialize the Inject marker.

        Args:
        ----
            value: The value to use if not injected

        """
        self.value = value


# Re-export the lifecycle scopes
SINGLETON = _LifecycleScope.SINGLETON
SCOPED = _LifecycleScope.SCOPED
TRANSIENT = _LifecycleScope.TRANSIENT


@stable
class LifecycleScopes:
    """
    Lifecycle scopes for services.

    This class defines the different lifecycle scopes for services:
    - SINGLETON: Service is created once per container
    - SCOPED: Service is created once per scope
    - TRANSIENT: Service is created each time it is resolved
    """

    SINGLETON = SINGLETON
    SCOPED = SCOPED
    TRANSIENT = TRANSIENT


@stable
class Scope(_Scope):
    """
    Scope for dependency injection.

    A scope manages the lifetime of services and provides a context
    for resolving scoped services.
    """


@stable
def inject(func):
    """
    Decorator to inject dependencies into a function.

    Args:
    ----
        func: The function to inject dependencies into

    Returns:
    -------
        A wrapper function that injects dependencies

    """
    return _inject(func)


@stable
def register(
    service_type=None,
    concrete_type=None,
    singleton=True,
    scope=None,
    provider=None,
    dependencies=None,
    **kwargs,
):
    """
    Decorator to register a class with the container.

    Args:
    ----
        service_type: The type to register
        concrete_type: The concrete type to instantiate
        singleton: Whether to register as singleton (deprecated, use scope instead)
        scope: The lifecycle scope for the registration
        provider: A provider instance to use for creating instances
        dependencies: Types that this service depends on for initialization
        **kwargs: Additional arguments to pass to the factory

    Returns:
    -------
        A decorator function

    """
    # Pass all arguments to the internal register function
    kwargs.update(
        {
            "singleton": singleton,
        }
    )
    if scope is not None:
        kwargs["scope"] = scope
    if provider is not None:
        kwargs["provider"] = provider
    if dependencies is not None:
        kwargs["dependencies"] = dependencies
    return _register(service_type, concrete_type, **kwargs)


@stable
def reset_container():
    """
    Reset the container to its default state.
    """
    return _reset_container()


# Container state management without global variables
from saplings._internal.container_state import ContainerState

# Replace global state with instance-based state
_container_state = ContainerState()


@stable
def configure_container(config, context_id: str = "default"):
    """
    Configure the container with the given configuration.

    Args:
    ----
        config: The configuration to use for the container.
        context_id: Context identifier for configuration isolation.

    """
    import logging

    logger = logging.getLogger(__name__)

    # Use context-based configuration instead of global state
    with _container_state.configuration_context(context_id) as is_first_config:
        if is_first_config:
            # Perform actual configuration
            if config is not None:
                container.register(config.__class__, instance=config)
                _container_state.set_configuration(config, context_id)

                # Configure services using the container_config module
                # Use a function to avoid circular imports
                _configure_services_with_config(config)

            logger.info(f"Container configured with dependency injection for context: {context_id}")
        else:
            # Update existing configuration
            if config is not None:
                # Update the configuration (container.register overwrites existing registrations)
                container.register(config.__class__, instance=config)
                _container_state.set_configuration(config, context_id)

            logger.debug(f"Container configuration updated for context: {context_id}")

    return container


@stable
def reset_container_config():
    """
    Reset the container configuration to its default state.
    """
    # Reset the default context
    reset_container_context("default")


def reset_container_context(context_id: str = "default"):
    """
    Reset container context safely.

    Args:
    ----
        context_id: Context identifier to reset

    """
    import logging

    logger = logging.getLogger(__name__)

    # Reset the context in container state
    _container_state.reset_context(context_id)

    # Reset the actual container
    reset_container()

    logger.info(f"Container context reset: {context_id}")


def _configure_services_with_config(config):
    """
    Configure services using the container_config module.

    This function is used to avoid circular imports between container.py and container_config.py.

    Args:
    ----
        config: The configuration to use for the container.

    """
    try:
        # Import here to avoid circular imports
        from saplings._internal.container_config import configure_services

        configure_services(config)
    except ImportError:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("container_config module not found, skipping service configuration")


# Re-export provider classes with stability annotations
Provider = beta(_Provider)
Provider.__doc__ = """
Base provider interface for dependency injection.

Providers are responsible for creating and managing instances of services.
This is the base interface that all providers must implement.
"""

FactoryProvider = beta(_FactoryProvider)
FactoryProvider.__doc__ = """
Provider that uses a factory function to create instances.

This provider uses a factory function to create instances of a service.
The factory function can take arguments that are resolved from the container.

Example:
```python
# Create a factory provider
provider = FactoryProvider(
    lambda db: UserService(database=db),
    db=Database
)

# Register with the container
container.register(UserService, factory=provider.provide)
```
"""

SingletonProvider = beta(_SingletonProvider)
SingletonProvider.__doc__ = """
Provider that ensures only one instance is created.

This provider wraps another provider and ensures that only one instance
is ever created, regardless of how many times it is resolved.

Example:
```python
# Create a singleton provider
provider = SingletonProvider(
    FactoryProvider(lambda: ExpensiveService())
)

# Register with the container
container.register(ExpensiveService, factory=provider.provide)
```
"""

ConfiguredProvider = beta(_ConfiguredProvider)
ConfiguredProvider.__doc__ = """
Provider that configures instances after creation.

This provider wraps another provider and applies configuration
to the instance after it is created.

Example:
```python
# Create a configured provider
provider = ConfiguredProvider(
    FactoryProvider(lambda: Database()),
    lambda db: db.configure(host="localhost", port=5432)
)

# Register with the container
container.register(Database, factory=provider.provide)
```
"""

LazyProvider = beta(_LazyProvider)
LazyProvider.__doc__ = """
Provider that defers creation until first access.

This provider wraps another provider and defers the creation of
the instance until it is first accessed.

Example:
```python
# Create a lazy provider
provider = LazyProvider(
    SingletonProvider(
        FactoryProvider(lambda: ExpensiveService())
    )
)

# Register with the container
container.register(ExpensiveService, factory=provider.provide)
```
"""

InitializableProvider = beta(_InitializableProvider)
InitializableProvider.__doc__ = """
Provider that supports initialization tracking.

This provider wraps another provider and adds initialization tracking.
It can be used to ensure that services are properly initialized before
they are used.

Example:
```python
# Create an initializable provider
provider = InitializableProvider(
    SingletonProvider(ServiceImpl()),
    dependencies=[Database, Logger]
)

# Register with the container
container.register(Service, provider=provider)
```
"""


# Re-export the initialization status enum
@stable
class InitializationStatus:
    """
    Initialization status for services.

    This enum defines the different initialization states for services:
    - UNINITIALIZED: Service has not been initialized
    - INITIALIZING: Service is currently being initialized
    - INITIALIZED: Service has been successfully initialized
    - FAILED: Service initialization failed
    """

    UNINITIALIZED = _InitializationStatus.UNINITIALIZED
    INITIALIZING = _InitializationStatus.INITIALIZING
    INITIALIZED = _InitializationStatus.INITIALIZED
    FAILED = _InitializationStatus.FAILED


__all__ = [
    # Container
    "Container",
    "container",
    "get_container",
    "inject",
    "register",
    "reset_container",
    "configure_container",
    "reset_container_config",
    "reset_container_context",
    # Inject
    "Inject",
    # Scopes
    "Scope",
    "LifecycleScopes",
    "SINGLETON",
    "SCOPED",
    "TRANSIENT",
    # Providers
    "Provider",
    "FactoryProvider",
    "SingletonProvider",
    "ConfiguredProvider",
    "LazyProvider",
    "InitializableProvider",
    # Initialization
    "InitializationStatus",
    # Service
    "Service",
]
