from __future__ import annotations

"""
Core event system for Saplings.

This module provides a centralized event system for cross-service communication,
helping to reduce direct dependencies between services and enabling a more
loosely coupled architecture.
"""

import asyncio
import logging
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CoreEventType(Enum):
    """Core event types for cross-service communication."""

    # Execution events
    EXECUTION_STARTED = auto()
    EXECUTION_COMPLETED = auto()
    EXECUTION_FAILED = auto()

    # Validation events
    VALIDATION_REQUESTED = auto()
    VALIDATION_COMPLETED = auto()
    VALIDATION_FAILED = auto()

    # Model events
    MODEL_INITIALIZED = auto()
    MODEL_GENERATED = auto()
    MODEL_FAILED = auto()

    # Judge events
    JUDGE_INITIALIZED = auto()
    JUDGE_REQUESTED = auto()
    JUDGE_COMPLETED = auto()
    JUDGE_FAILED = auto()

    # Error events
    ERROR_OCCURRED = auto()

    # Lifecycle events
    SERVICE_INITIALIZED = auto()
    SERVICE_DISPOSED = auto()
    SERVICE_STATE_CHANGED = auto()
    SERVICE_INITIALIZING = auto()
    SERVICE_READY = auto()
    SERVICE_SHUTTING_DOWN = auto()

    # Dependency events
    DEPENDENCY_READY = auto()
    DEPENDENCY_FAILED = auto()
    DEPENDENCY_WAITING = auto()
    DEPENDENCY_CYCLE_DETECTED = auto()

    # Mediator events
    SERVICE_REQUEST_HANDLED = auto()
    SERVICE_REQUEST_FAILED = auto()


class CoreEvent:
    """Event data structure for core events."""

    def __init__(
        self,
        event_type: CoreEventType,
        data: Dict[str, Any],
        source: str,
        trace_id: Optional[str] = None,
    ):
        """
        Initialize a core event.

        Args:
        ----
            event_type: Type of the event
            data: Data associated with the event
            source: Source of the event (typically service name)
            trace_id: Optional trace ID for correlation

        """
        self.event_type = event_type
        self.data = data
        self.source = source
        self.trace_id = trace_id
        self.timestamp = time.time()

    def __str__(self) -> str:
        """Return string representation of the event."""
        return f"CoreEvent({self.event_type}, source={self.source}, trace_id={self.trace_id})"


EventHandler = Callable[[CoreEvent], None]
AsyncEventHandler = Callable[[CoreEvent], Any]


class CoreEventBus:
    """
    Event bus for core service communication.

    This class provides a centralized event bus for services to communicate
    without direct dependencies, helping to reduce circular dependencies.
    """

    def __init__(self):
        """Initialize the event bus."""
        self._handlers: Dict[CoreEventType, List[EventHandler]] = {}
        self._async_handlers: Dict[CoreEventType, List[AsyncEventHandler]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._is_processing = False
        logger.info("CoreEventBus initialized")

    def subscribe(
        self,
        event_type: Union[CoreEventType, List[CoreEventType]],
        handler: Union[EventHandler, AsyncEventHandler],
        is_async: bool = False,
    ) -> None:
        """
        Subscribe to an event type.

        Args:
        ----
            event_type: Event type or list of event types to subscribe to
            handler: Function to call when the event occurs
            is_async: Whether the handler is asynchronous

        """
        if isinstance(event_type, list):
            for et in event_type:
                self.subscribe(et, handler, is_async)
            return

        if is_async:
            if event_type not in self._async_handlers:
                self._async_handlers[event_type] = []
            self._async_handlers[event_type].append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

        logger.debug(f"Subscribed to {event_type}: {handler.__name__}")

    def unsubscribe(
        self,
        event_type: Union[CoreEventType, List[CoreEventType]],
        handler: Union[EventHandler, AsyncEventHandler],
        is_async: bool = False,
    ) -> None:
        """
        Unsubscribe from an event type.

        Args:
        ----
            event_type: Event type or list of event types to unsubscribe from
            handler: Handler to remove
            is_async: Whether the handler is asynchronous

        """
        if isinstance(event_type, list):
            for et in event_type:
                self.unsubscribe(et, handler, is_async)
            return

        if is_async:
            if event_type in self._async_handlers and handler in self._async_handlers[event_type]:
                self._async_handlers[event_type].remove(handler)
        elif event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

        logger.debug(f"Unsubscribed from {event_type}: {handler.__name__}")

    def publish(self, event: CoreEvent) -> None:
        """
        Publish an event to the bus.

        Args:
        ----
            event: Event to publish

        """
        # Add the event to the queue
        self._event_queue.put_nowait(event)

        # Start processing if not already running
        if not self._is_processing:
            self._start_processing()

        logger.debug(f"Published event: {event}")

    def _start_processing(self) -> None:
        """Start processing events from the queue."""
        if self._processing_task is None or self._processing_task.done():
            loop = asyncio.get_event_loop()
            self._processing_task = loop.create_task(self._process_events())
            self._is_processing = True

    async def _process_events(self) -> None:
        """Process events from the queue."""
        try:
            while not self._event_queue.empty():
                # Get the next event
                event = await self._event_queue.get()

                # Call synchronous handlers
                self._call_sync_handlers(event)

                # Call asynchronous handlers
                await self._call_async_handlers(event)

                # Mark the event as processed
                self._event_queue.task_done()
        except Exception as e:
            logger.exception(f"Error processing events: {e}")
        finally:
            self._is_processing = False

    def _call_sync_handlers(self, event: CoreEvent) -> None:
        """
        Call synchronous handlers for an event.

        Args:
        ----
            event: Event to process

        """
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.exception(f"Error in handler {handler.__name__} for {event.event_type}: {e}")

    async def _call_async_handlers(self, event: CoreEvent) -> None:
        """
        Call asynchronous handlers for an event.

        Args:
        ----
            event: Event to process

        """
        for handler in self._async_handlers.get(event.event_type, []):
            try:
                await handler(event)
            except Exception as e:
                logger.exception(
                    f"Error in async handler {handler.__name__} for {event.event_type}: {e}"
                )

    def clear(self) -> None:
        """Clear all handlers."""
        self._handlers.clear()
        self._async_handlers.clear()
        logger.info("Cleared all event handlers")


# Global event bus instance
_event_bus: Optional[CoreEventBus] = None


def get_event_bus() -> CoreEventBus:
    """
    Get the global event bus instance.

    Returns
    -------
        CoreEventBus: The global event bus instance

    """
    global _event_bus
    if _event_bus is None:
        _event_bus = CoreEventBus()
    return _event_bus
