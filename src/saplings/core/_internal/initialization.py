from __future__ import annotations

"""
Initialization utilities for Saplings services.

This module provides utility functions for initializing services and managing
their dependencies.
"""

import logging
from typing import Any, List, Optional, TypeVar

from saplings.core._internal.exceptions import InitializationError
from saplings.core._internal.service_registry import get_service_registry
from saplings.core.events import CoreEvent, CoreEventType, get_event_bus
from saplings.core.lifecycle import ServiceLifecycle, ServiceState

logger = logging.getLogger(__name__)

T = TypeVar("T")


def initialize_service(
    service_instance: Any,
    service_name: Optional[str] = None,
    dependencies: Optional[List[Any]] = None,
) -> ServiceLifecycle:
    """
    Initialize a service and register it with the service registry.

    This function creates a lifecycle object for the service, registers it with
    the service registry, and transitions it to the INITIALIZING state. It also
    registers any dependencies and sets up event-based initialization.

    Args:
    ----
        service_instance: Service instance to initialize
        service_name: Optional name for the service (defaults to class name)
        dependencies: Optional list of dependency service instances

    Returns:
    -------
        ServiceLifecycle: Lifecycle object for the service

    """
    # Get the service name
    if service_name is None:
        service_name = service_instance.__class__.__name__

    # Get the service registry
    registry = get_service_registry()

    # Register the service
    lifecycle = registry.register_service(service_name)

    # Attach the lifecycle to the service instance
    service_instance._lifecycle = lifecycle

    # Register dependencies if provided
    if dependencies:
        for dependency in dependencies:
            register_dependency(service_instance, dependency)

    # Transition to INITIALIZING state
    # This will automatically handle dependency waiting
    lifecycle.transition_to(ServiceState.INITIALIZING)

    # Publish event
    get_event_bus().publish(
        CoreEvent(
            event_type=CoreEventType.SERVICE_INITIALIZING,
            data={"service_name": service_name},
            source=lifecycle.service_name,  # Use the lifecycle's service_name which is guaranteed to be a string
        )
    )

    logger.info(f"Initialized service: {service_name}")
    return lifecycle


def mark_service_ready(service_instance: Any, error: Optional[Exception] = None) -> None:
    """
    Mark a service as ready or failed.

    This function transitions the service to the READY state or to DISPOSED if an error is provided.

    Args:
    ----
        service_instance: Service instance to mark as ready
        error: Optional error that prevented the service from being ready

    Raises:
    ------
        InitializationError: If the service is not in the INITIALIZING state

    """
    # Get the lifecycle object
    lifecycle = getattr(service_instance, "_lifecycle", None)
    if lifecycle is None:
        raise InitializationError(
            f"Service {service_instance.__class__.__name__} has not been initialized"
        )

    if error:
        # If there was an error, transition to DISPOSED state
        lifecycle.transition_to(ServiceState.DISPOSED, error=error)
        logger.error(f"Service {lifecycle.service_name} failed to initialize: {error}")
        return

    # Check if all dependencies are ready
    if not lifecycle.all_dependencies_ready:
        # This shouldn't happen if using event-based initialization correctly
        # but we'll handle it gracefully
        missing_deps = lifecycle.dependencies - lifecycle.ready_dependencies
        logger.warning(
            f"Service {lifecycle.service_name} is being marked as ready but "
            f"dependencies are not ready: {missing_deps}"
        )

    # Transition to READY state
    lifecycle.transition_to(ServiceState.READY)

    # Publish event
    get_event_bus().publish(
        CoreEvent(
            event_type=CoreEventType.SERVICE_READY,
            data={"service_name": lifecycle.service_name},
            source=lifecycle.service_name,
        )
    )

    logger.info(f"Service {lifecycle.service_name} is ready")


def shutdown_service(service_instance: Any) -> None:
    """
    Shut down a service.

    This function transitions the service to the SHUTTING_DOWN state.

    Args:
    ----
        service_instance: Service instance to shut down

    Raises:
    ------
        InitializationError: If the service is not in the READY state

    """
    # Get the lifecycle object
    lifecycle = getattr(service_instance, "_lifecycle", None)
    if lifecycle is None:
        raise InitializationError(
            f"Service {service_instance.__class__.__name__} has not been initialized"
        )

    # Transition to SHUTTING_DOWN state
    lifecycle.transition_to(ServiceState.SHUTTING_DOWN)

    # Publish event
    get_event_bus().publish(
        CoreEvent(
            event_type=CoreEventType.SERVICE_SHUTTING_DOWN,
            data={"service_name": lifecycle.service_name},
            source=lifecycle.service_name,
        )
    )

    logger.info(f"Service {lifecycle.service_name} is shutting down")


def dispose_service(service_instance: Any) -> None:
    """
    Dispose of a service.

    This function transitions the service to the DISPOSED state.

    Args:
    ----
        service_instance: Service instance to dispose

    Raises:
    ------
        InitializationError: If the service is not in the SHUTTING_DOWN state

    """
    # Get the lifecycle object
    lifecycle = getattr(service_instance, "_lifecycle", None)
    if lifecycle is None:
        raise InitializationError(
            f"Service {service_instance.__class__.__name__} has not been initialized"
        )

    # Transition to DISPOSED state
    lifecycle.transition_to(ServiceState.DISPOSED)

    # Publish event
    get_event_bus().publish(
        CoreEvent(
            event_type=CoreEventType.SERVICE_DISPOSED,
            data={"service_name": lifecycle.service_name},
            source=lifecycle.service_name,
        )
    )

    logger.info(f"Service {lifecycle.service_name} has been disposed")


def register_dependency(
    service_instance: Any, dependency_instance: Any, dependency_name: Optional[str] = None
) -> None:
    """
    Register a dependency between services.

    This function adds a dependency relationship between two services.

    Args:
    ----
        service_instance: Dependent service instance
        dependency_instance: Dependency service instance
        dependency_name: Optional name for the dependency (defaults to class name)

    Raises:
    ------
        InitializationError: If either service has not been initialized

    """
    # Get the lifecycle objects
    service_lifecycle = getattr(service_instance, "_lifecycle", None)
    dependency_lifecycle = getattr(dependency_instance, "_lifecycle", None)

    if service_lifecycle is None:
        raise InitializationError(
            f"Service {service_instance.__class__.__name__} has not been initialized"
        )
    if dependency_lifecycle is None:
        raise InitializationError(
            f"Dependency {dependency_instance.__class__.__name__} has not been initialized"
        )

    # Get the dependency name
    if dependency_name is None:
        dependency_name = dependency_instance.__class__.__name__

    # Register the dependency
    registry = get_service_registry()
    registry.add_dependency(service_lifecycle.service_name, dependency_lifecycle.service_name)

    logger.info(
        f"Registered dependency: {service_lifecycle.service_name} -> {dependency_lifecycle.service_name}"
    )


def get_initialization_order() -> List[str]:
    """
    Get the order in which services should be initialized.

    Returns
    -------
        List[str]: List of service names in initialization order

    """
    registry = get_service_registry()
    return registry.get_initialization_order()


def get_service_state(service_instance: Any) -> Optional[ServiceState]:
    """
    Get the state of a service.

    Args:
    ----
        service_instance: Service instance

    Returns:
    -------
        Optional[ServiceState]: State of the service, or None if not initialized

    """
    lifecycle = getattr(service_instance, "_lifecycle", None)
    if lifecycle is None:
        return None
    return lifecycle.state
