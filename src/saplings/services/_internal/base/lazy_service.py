from __future__ import annotations

"""
Lazy Service module for Saplings.

This module provides a base class for services that support lazy initialization.
"""

import asyncio
import logging
import threading
from enum import Enum
from typing import Any, Dict, Optional, Set, TypeVar

from saplings.core._internal.exceptions import InitializationError, OperationTimeoutError
from saplings.core._internal.resilience.resilience import with_timeout

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Enum representing the status of a service."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


T = TypeVar("T", bound="LazyService")


class LazyService:
    """
    Base class for services that support lazy initialization.

    This class provides methods for tracking initialization status,
    lazy initialization of dependencies, and graceful shutdown.
    """

    def __init__(self) -> None:
        """Initialize the lazy service."""
        self._status = ServiceStatus.UNINITIALIZED
        self._status_lock = threading.RLock()
        self._initialization_event = asyncio.Event()
        self._initialization_error: Optional[Exception] = None
        self._dependencies: Dict[str, Any] = {}
        self._initialized_dependencies: Set[str] = set()

    @property
    def status(self) -> ServiceStatus:
        """
        Get the current status of the service.

        Returns
        -------
            ServiceStatus: Current status

        """
        with self._status_lock:
            return self._status

    def register_dependency(self, name: str, dependency: Any) -> None:
        """
        Register a dependency for the service.

        Args:
        ----
            name: Name of the dependency
            dependency: The dependency instance

        """
        self._dependencies[name] = dependency

    async def initialize(self, timeout: Optional[float] = None) -> None:
        """
        Initialize the service and its dependencies.

        This method ensures that the service is initialized only once,
        even if called multiple times concurrently.

        Args:
        ----
            timeout: Optional timeout in seconds

        Raises:
        ------
            InitializationError: If initialization fails
            OperationTimeoutError: If initialization times out

        """
        # Check if already initialized
        with self._status_lock:
            if self._status == ServiceStatus.READY:
                return
            elif self._status == ServiceStatus.FAILED:
                if self._initialization_error:
                    raise InitializationError(
                        f"Service initialization previously failed: {self._initialization_error}"
                    ) from self._initialization_error
                else:
                    raise InitializationError("Service initialization previously failed")
            elif self._status == ServiceStatus.INITIALIZING:
                # Another thread is initializing, wait for it
                pass
            elif self._status in (ServiceStatus.SHUTTING_DOWN, ServiceStatus.SHUTDOWN):
                raise InitializationError(
                    "Cannot initialize a service that is shutting down or shutdown"
                )
            else:
                # Set status to initializing
                self._status = ServiceStatus.INITIALIZING

        # Wait for initialization to complete or timeout
        try:
            # If we're the first thread to initialize, do the initialization
            if not self._initialization_event.is_set():
                try:
                    # Initialize dependencies first
                    await self._initialize_dependencies(timeout)

                    # Initialize the service
                    await with_timeout(
                        self._initialize_impl(),
                        timeout=timeout,
                        operation_name=f"initialize_{self.__class__.__name__}",
                    )

                    # Set status to ready
                    with self._status_lock:
                        self._status = ServiceStatus.READY

                    # Signal that initialization is complete
                    self._initialization_event.set()
                except Exception as e:
                    # Set status to failed
                    with self._status_lock:
                        self._status = ServiceStatus.FAILED
                        self._initialization_error = e

                    # Signal that initialization is complete (with error)
                    self._initialization_event.set()

                    # Re-raise the exception
                    raise
            else:
                # Wait for initialization to complete
                await with_timeout(
                    self._initialization_event.wait(),
                    timeout=timeout,
                    operation_name=f"wait_for_initialize_{self.__class__.__name__}",
                )

                # Check if initialization failed
                with self._status_lock:
                    if self._status == ServiceStatus.FAILED:
                        if self._initialization_error:
                            raise InitializationError(
                                f"Service initialization failed: {self._initialization_error}"
                            ) from self._initialization_error
                        else:
                            raise InitializationError("Service initialization failed")
        except asyncio.TimeoutError:
            raise OperationTimeoutError(f"Initialization of {self.__class__.__name__} timed out")

    async def _initialize_dependencies(self, timeout: Optional[float] = None) -> None:
        """
        Initialize all dependencies.

        Args:
        ----
            timeout: Optional timeout in seconds

        Raises:
        ------
            InitializationError: If dependency initialization fails

        """
        for name, dependency in self._dependencies.items():
            if name in self._initialized_dependencies:
                continue

            if isinstance(dependency, LazyService):
                try:
                    await dependency.initialize(timeout=timeout)
                    self._initialized_dependencies.add(name)
                except Exception as e:
                    raise InitializationError(
                        f"Failed to initialize dependency '{name}' for {self.__class__.__name__}"
                    ) from e
            else:
                # Non-lazy dependencies are considered already initialized
                self._initialized_dependencies.add(name)

    async def _initialize_impl(self) -> None:
        """
        Implementation of service initialization.

        This method should be overridden by subclasses to provide
        service-specific initialization logic.

        Raises
        ------
            NotImplementedError: If not overridden by subclass

        """
        # Default implementation does nothing

    async def shutdown(self, timeout: Optional[float] = None) -> None:
        """
        Shut down the service.

        Args:
        ----
            timeout: Optional timeout in seconds

        Raises:
        ------
            OperationTimeoutError: If shutdown times out

        """
        with self._status_lock:
            if self._status in (ServiceStatus.SHUTTING_DOWN, ServiceStatus.SHUTDOWN):
                return
            self._status = ServiceStatus.SHUTTING_DOWN

        try:
            # Shut down the service
            await with_timeout(
                self._shutdown_impl(),
                timeout=timeout,
                operation_name=f"shutdown_{self.__class__.__name__}",
            )

            # Set status to shutdown
            with self._status_lock:
                self._status = ServiceStatus.SHUTDOWN
        except asyncio.TimeoutError:
            raise OperationTimeoutError(f"Shutdown of {self.__class__.__name__} timed out")
        except Exception as e:
            logger.error(f"Error during shutdown of {self.__class__.__name__}: {e}")
            # Still mark as shutdown
            with self._status_lock:
                self._status = ServiceStatus.SHUTDOWN
            raise

    async def _shutdown_impl(self) -> None:
        """
        Implementation of service shutdown.

        This method should be overridden by subclasses to provide
        service-specific shutdown logic.
        """
        # Default implementation does nothing
