from __future__ import annotations

"""
Service lifecycle management for Saplings.

This module provides classes and utilities for managing the lifecycle of services,
including state tracking, validation, and cleanup.
"""

import enum
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Set

from saplings.core._internal.exceptions import InitializationError
from saplings.core.events import CoreEvent, CoreEventType, get_event_bus

logger = logging.getLogger(__name__)


class ServiceState(enum.Enum):
    """
    Service lifecycle states.

    - UNINITIALIZED: Service has been created but not initialized
    - INITIALIZING: Service is in the process of being initialized
    - READY: Service is fully initialized and ready for use
    - SHUTTING_DOWN: Service is in the process of shutting down
    - DISPOSED: Service has been disposed and should not be used
    """

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"
    DISPOSED = "disposed"


class ServiceLifecycle:
    """
    Service lifecycle management.

    This class provides methods for tracking and validating the lifecycle state
    of a service, as well as handling state transitions. It supports event-based
    initialization and dependency tracking.
    """

    def __init__(self, service_name: str) -> None:
        """
        Initialize the service lifecycle.

        Args:
        ----
            service_name: Name of the service

        """
        self._service_name = service_name
        self._state = ServiceState.UNINITIALIZED
        self._state_lock = threading.RLock()
        self._event_bus = get_event_bus()
        self._dependencies: Set[str] = set()
        self._dependents: Set[str] = set()
        self._ready_dependencies: Set[str] = set()
        self._failed_dependencies: Set[str] = set()
        self._initialization_error: Optional[Exception] = None
        self._dependency_handlers: Dict[str, List[Callable]] = {}
        logger.debug(f"ServiceLifecycle initialized for {service_name}")

    @property
    def state(self) -> ServiceState:
        """Get the current state of the service."""
        with self._state_lock:
            return self._state

    @property
    def service_name(self) -> str:
        """Get the name of the service."""
        return self._service_name

    @property
    def dependencies(self) -> Set[str]:
        """Get the dependencies of the service."""
        return self._dependencies.copy()

    @property
    def dependents(self) -> Set[str]:
        """Get the dependents of the service."""
        return self._dependents.copy()

    @property
    def ready_dependencies(self) -> Set[str]:
        """Get the dependencies that are ready."""
        return self._ready_dependencies.copy()

    @property
    def failed_dependencies(self) -> Set[str]:
        """Get the dependencies that failed to initialize."""
        return self._failed_dependencies.copy()

    @property
    def initialization_error(self) -> Optional[Exception]:
        """Get the initialization error, if any."""
        return self._initialization_error

    @property
    def all_dependencies_ready(self) -> bool:
        """Check if all dependencies are ready."""
        return len(self._ready_dependencies) == len(self._dependencies)

    @property
    def any_dependencies_failed(self) -> bool:
        """Check if any dependencies failed to initialize."""
        return len(self._failed_dependencies) > 0

    def add_dependency(self, service_name: str) -> None:
        """
        Add a dependency to the service.

        Args:
        ----
            service_name: Name of the dependency service

        """
        with self._state_lock:
            if service_name not in self._dependencies:
                self._dependencies.add(service_name)
                # Subscribe to dependency events
                self._subscribe_to_dependency_events(service_name)
                # Publish event
                self._event_bus.publish(
                    CoreEvent(
                        event_type=CoreEventType.DEPENDENCY_WAITING,
                        data={
                            "service_name": self._service_name,
                            "dependency_name": service_name,
                        },
                        source=self._service_name,
                    )
                )
                logger.debug(f"Added dependency {service_name} to {self._service_name}")

    def add_dependent(self, service_name: str) -> None:
        """
        Add a dependent to the service.

        Args:
        ----
            service_name: Name of the dependent service

        """
        with self._state_lock:
            if service_name not in self._dependents:
                self._dependents.add(service_name)
                logger.debug(f"Added dependent {service_name} to {self._service_name}")

    def mark_dependency_ready(self, dependency_name: str) -> None:
        """
        Mark a dependency as ready.

        Args:
        ----
            dependency_name: Name of the dependency service

        """
        with self._state_lock:
            if dependency_name in self._dependencies:
                self._ready_dependencies.add(dependency_name)
                logger.debug(f"Dependency {dependency_name} is ready for {self._service_name}")

                # Check if all dependencies are ready
                if self.all_dependencies_ready:
                    logger.info(f"All dependencies ready for {self._service_name}")
                    # Execute any registered handlers
                    for handler in self._dependency_handlers.get("all_ready", []):
                        try:
                            handler()
                        except Exception as e:
                            logger.exception(
                                f"Error in dependency handler for {self._service_name}: {e}"
                            )

    def mark_dependency_failed(
        self, dependency_name: str, error: Optional[Exception] = None
    ) -> None:
        """
        Mark a dependency as failed.

        Args:
        ----
            dependency_name: Name of the dependency service
            error: Optional error that caused the failure

        """
        with self._state_lock:
            if dependency_name in self._dependencies:
                self._failed_dependencies.add(dependency_name)
                logger.error(
                    f"Dependency {dependency_name} failed for {self._service_name}: {error}"
                )

                # Execute any registered handlers
                for handler in self._dependency_handlers.get("any_failed", []):
                    try:
                        handler(dependency_name, error)
                    except Exception as e:
                        logger.exception(
                            f"Error in dependency failure handler for {self._service_name}: {e}"
                        )

    def on_all_dependencies_ready(self, handler: Callable) -> None:
        """
        Register a handler to be called when all dependencies are ready.

        Args:
        ----
            handler: Function to call when all dependencies are ready

        """
        with self._state_lock:
            if "all_ready" not in self._dependency_handlers:
                self._dependency_handlers["all_ready"] = []
            self._dependency_handlers["all_ready"].append(handler)

            # If all dependencies are already ready, call the handler immediately
            if self.all_dependencies_ready:
                try:
                    handler()
                except Exception as e:
                    logger.exception(f"Error in dependency handler for {self._service_name}: {e}")

    def on_any_dependency_failed(self, handler: Callable[[str, Optional[Exception]], None]) -> None:
        """
        Register a handler to be called when any dependency fails.

        Args:
        ----
            handler: Function to call when a dependency fails

        """
        with self._state_lock:
            if "any_failed" not in self._dependency_handlers:
                self._dependency_handlers["any_failed"] = []
            self._dependency_handlers["any_failed"].append(handler)

            # If any dependencies have already failed, call the handler immediately
            if self.any_dependencies_failed:
                for failed_dep in self._failed_dependencies:
                    try:
                        handler(failed_dep, self._initialization_error)
                    except Exception as e:
                        logger.exception(
                            f"Error in dependency failure handler for {self._service_name}: {e}"
                        )

    def _subscribe_to_dependency_events(self, dependency_name: str) -> None:
        """
        Subscribe to events from a dependency.

        Args:
        ----
            dependency_name: Name of the dependency service

        """

        # Subscribe to state change events
        def handle_state_change(event: CoreEvent) -> None:
            if event.source == dependency_name:
                new_state = event.data.get("new_state")
                if new_state == ServiceState.READY.value:
                    self.mark_dependency_ready(dependency_name)
                elif new_state == ServiceState.DISPOSED.value and event.data.get("error"):
                    self.mark_dependency_failed(dependency_name, event.data.get("error"))

        self._event_bus.subscribe(CoreEventType.SERVICE_STATE_CHANGED, handle_state_change)

    def transition_to(self, state: ServiceState, error: Optional[Exception] = None) -> None:
        """
        Transition the service to a new state.

        Args:
        ----
            state: New state to transition to
            error: Optional error that caused the transition (for failure cases)

        Raises:
        ------
            InitializationError: If the transition is invalid

        """
        with self._state_lock:
            # Check if the transition is valid
            if not self._is_valid_transition(self._state, state):
                raise InitializationError(
                    f"Invalid state transition for {self._service_name}: "
                    f"{self._state.value} -> {state.value}"
                )

            # For INITIALIZING state, check if dependencies are ready
            if state == ServiceState.INITIALIZING and self._dependencies:
                if not self.all_dependencies_ready:
                    # Register a handler to auto-transition when dependencies are ready
                    self.on_all_dependencies_ready(lambda: self._continue_initialization())
                    # Register a handler for dependency failures
                    self.on_any_dependency_failed(
                        lambda dep_name, err: self._handle_dependency_failure(dep_name, err)
                    )
                    logger.info(
                        f"Service {self._service_name} waiting for dependencies before initializing"
                    )
                    # Still update the state to INITIALIZING
                    old_state = self._state
                    self._state = state

                    # Publish event with dependency information
                    self._event_bus.publish(
                        CoreEvent(
                            event_type=CoreEventType.SERVICE_STATE_CHANGED,
                            data={
                                "service_name": self._service_name,
                                "old_state": old_state.value,
                                "new_state": state.value,
                                "waiting_for_dependencies": list(
                                    self._dependencies - self._ready_dependencies
                                ),
                            },
                            source=self._service_name,
                        )
                    )
                    return

            # Store initialization error if provided
            if error is not None:
                self._initialization_error = error

            # Update the state
            old_state = self._state
            self._state = state

            # Log the transition
            if error:
                logger.warning(
                    f"Service {self._service_name} transitioned from {old_state.value} to {state.value} "
                    f"with error: {error}"
                )
            else:
                logger.info(
                    f"Service {self._service_name} transitioned from {old_state.value} to {state.value}"
                )

            # Publish event
            event_data = {
                "service_name": self._service_name,
                "old_state": old_state.value,
                "new_state": state.value,
            }

            # Add error information if available
            if error:
                event_data["error"] = str(error)
                event_data["error_type"] = type(error).__name__

            # Add dependency information if relevant
            if self._dependencies:
                event_data["dependency_total"] = str(len(self._dependencies))
                event_data["dependency_ready"] = str(len(self._ready_dependencies))
                event_data["dependency_failed"] = str(len(self._failed_dependencies))
                event_data["dependency_pending"] = str(
                    len(self._dependencies - self._ready_dependencies - self._failed_dependencies)
                )

            self._event_bus.publish(
                CoreEvent(
                    event_type=CoreEventType.SERVICE_STATE_CHANGED,
                    data=event_data,
                    source=self._service_name,
                )
            )

            # If transitioning to READY, publish a specific event for dependencies
            if state == ServiceState.READY:
                self._event_bus.publish(
                    CoreEvent(
                        event_type=CoreEventType.DEPENDENCY_READY,
                        data={"service_name": self._service_name},
                        source=self._service_name,
                    )
                )

            # If transitioning to DISPOSED with an error, publish a failure event
            if state == ServiceState.DISPOSED and error:
                self._event_bus.publish(
                    CoreEvent(
                        event_type=CoreEventType.DEPENDENCY_FAILED,
                        data={
                            "service_name": self._service_name,
                            "error": str(error),
                            "error_type": type(error).__name__,
                        },
                        source=self._service_name,
                    )
                )

    def _continue_initialization(self) -> None:
        """
        Continue initialization after dependencies are ready.

        This method is called automatically when all dependencies are ready.
        """
        logger.info(f"All dependencies ready, continuing initialization of {self._service_name}")
        # The service is already in INITIALIZING state, so we don't need to transition
        # This is just a notification that dependencies are ready
        self._event_bus.publish(
            CoreEvent(
                event_type=CoreEventType.SERVICE_STATE_CHANGED,
                data={
                    "service_name": self._service_name,
                    "old_state": ServiceState.INITIALIZING.value,
                    "new_state": ServiceState.INITIALIZING.value,
                    "dependencies_ready": True,
                },
                source=self._service_name,
            )
        )

    def _handle_dependency_failure(self, dependency_name: str, error: Optional[Exception]) -> None:
        """
        Handle a dependency failure.

        This method is called automatically when a dependency fails.

        Args:
        ----
            dependency_name: Name of the failed dependency
            error: Error that caused the failure

        """
        logger.error(
            f"Dependency {dependency_name} failed, cannot initialize {self._service_name}: {error}"
        )
        # Transition to DISPOSED state with the error
        self.transition_to(
            ServiceState.DISPOSED,
            error=InitializationError(
                f"Cannot initialize {self._service_name} due to failed dependency {dependency_name}: {error}"
            ),
        )

    def validate_state(self, required_state: ServiceState | List[ServiceState]) -> None:
        """
        Validate that the service is in the required state.

        Args:
        ----
            required_state: Required state or list of states

        Raises:
        ------
            InitializationError: If the service is not in the required state

        """
        with self._state_lock:
            if isinstance(required_state, list):
                if self._state not in required_state:
                    states_str = ", ".join(s.value for s in required_state)
                    raise InitializationError(
                        f"Service {self._service_name} is in state {self._state.value}, "
                        f"but one of [{states_str}] is required"
                    )
            elif self._state != required_state:
                raise InitializationError(
                    f"Service {self._service_name} is in state {self._state.value}, "
                    f"but {required_state.value} is required"
                )

    def _is_valid_transition(self, from_state: ServiceState, to_state: ServiceState) -> bool:
        """
        Check if a state transition is valid.

        Args:
        ----
            from_state: Current state
            to_state: Target state

        Returns:
        -------
            bool: True if the transition is valid, False otherwise

        """
        # Define valid transitions
        valid_transitions = {
            ServiceState.UNINITIALIZED: {ServiceState.INITIALIZING, ServiceState.DISPOSED},
            ServiceState.INITIALIZING: {ServiceState.READY, ServiceState.DISPOSED},
            ServiceState.READY: {ServiceState.SHUTTING_DOWN, ServiceState.DISPOSED},
            ServiceState.SHUTTING_DOWN: {ServiceState.DISPOSED},
            ServiceState.DISPOSED: set(),  # No transitions from DISPOSED
        }

        return to_state in valid_transitions.get(from_state, set())


def validate_service_state(
    required_state: ServiceState | List[ServiceState],
) -> Callable:
    """
    Decorator to validate service state before method execution.

    Args:
    ----
        required_state: Required state or list of states

    Returns:
    -------
        Callable: Decorator function

    """

    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            # Get the lifecycle from the service
            lifecycle = getattr(self, "_lifecycle", None)
            if lifecycle is not None:
                lifecycle.validate_state(required_state)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
