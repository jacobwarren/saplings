from __future__ import annotations

"""
Mediator module for cross-service communication.

This module provides a mediator service that facilitates communication between
services without creating direct dependencies. It follows the Mediator pattern
to reduce coupling between components.
"""

import logging
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

from saplings.core.events import CoreEvent, CoreEventType, get_event_bus

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceRequest:
    """Base class for service requests."""

    def __init__(self, request_type: str, data: Dict[str, Any], source: str):
        """
        Initialize a service request.

        Args:
        ----
            request_type: Type of request
            data: Request data
            source: Source of the request

        """
        self.request_type = request_type
        self.data = data
        self.source = source


class ServiceResponse:
    """Base class for service responses."""

    def __init__(self, success: bool, data: Dict[str, Any], error: Optional[str] = None):
        """
        Initialize a service response.

        Args:
        ----
            success: Whether the request was successful
            data: Response data
            error: Optional error message

        """
        self.success = success
        self.data = data
        self.error = error


class IServiceMediator(Protocol):
    """Interface for service mediator."""

    def register_handler(
        self, request_type: str, handler: Callable[[ServiceRequest], ServiceResponse]
    ) -> None:
        """
        Register a handler for a request type.

        Args:
        ----
            request_type: Type of request to handle
            handler: Handler function

        """
        ...

    def send(self, request: ServiceRequest) -> ServiceResponse:
        """
        Send a request to the appropriate handler.

        Args:
        ----
            request: Request to send

        Returns:
        -------
            Response from the handler

        """
        ...


class ServiceMediator(IServiceMediator):
    """
    Service mediator for cross-service communication.

    This class implements the Mediator pattern to facilitate communication
    between services without creating direct dependencies.
    """

    _instance: Optional[ServiceMediator] = None

    def __new__(cls) -> ServiceMediator:
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(ServiceMediator, cls).__new__(cls)
            cls._instance._handlers = {}
            cls._instance._event_bus = get_event_bus()
            logger.info("ServiceMediator initialized")
        return cls._instance

    def register_handler(
        self, request_type: str, handler: Callable[[ServiceRequest], ServiceResponse]
    ) -> None:
        """
        Register a handler for a request type.

        Args:
        ----
            request_type: Type of request to handle
            handler: Handler function

        """
        self._handlers[request_type] = handler
        logger.debug(f"Registered handler for request type: {request_type}")

    def send(self, request: ServiceRequest) -> ServiceResponse:
        """
        Send a request to the appropriate handler.

        Args:
        ----
            request: Request to send

        Returns:
        -------
            Response from the handler

        Raises:
        ------
            ValueError: If no handler is registered for the request type

        """
        if request.request_type not in self._handlers:
            error_msg = f"No handler registered for request type: {request.request_type}"
            logger.error(error_msg)
            return ServiceResponse(success=False, data={}, error=error_msg)

        try:
            # Get the handler and call it
            handler = self._handlers[request.request_type]
            response = handler(request)

            # Publish event for monitoring
            self._event_bus.publish(
                CoreEvent(
                    event_type=CoreEventType.SERVICE_REQUEST_HANDLED,
                    data={
                        "request_type": request.request_type,
                        "source": request.source,
                        "success": response.success,
                    },
                    source="ServiceMediator",
                )
            )

            return response
        except Exception as e:
            error_msg = f"Error handling request: {e}"
            logger.exception(error_msg)
            return ServiceResponse(success=False, data={}, error=error_msg)


def get_service_mediator() -> ServiceMediator:
    """
    Get the service mediator instance.

    Returns
    -------
        ServiceMediator: The service mediator instance

    """
    return ServiceMediator()
