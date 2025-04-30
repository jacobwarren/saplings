"""
Event system for Saplings.

This module provides an event system for cross-component communication.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events in the system."""

    TOOL_LOADED = "tool_loaded"
    TOOL_UNLOADED = "tool_unloaded"
    TOOL_UPDATED = "tool_updated"
    TOOL_EXECUTED = "tool_executed"
    PLAN_CREATED = "plan_created"
    PLAN_EXECUTED = "plan_executed"
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"
    CHANNEL_CREATED = "channel_created"
    CHANNEL_DELETED = "channel_deleted"
    MESSAGE_SENT = "message_sent"
    NEGOTIATION_STARTED = "negotiation_started"
    NEGOTIATION_COMPLETED = "negotiation_completed"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class Event:
    """Event data structure."""

    type: EventType
    """Type of the event."""

    source: str
    """Source of the event."""

    data: Dict[str, Any] = field(default_factory=dict)
    """Data associated with the event."""

    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    """Timestamp of the event."""


EventListener = Callable[[Event], None]
AsyncEventListener = Callable[[Event], Any]


class EventSystem:
    """
    Event system for cross-component communication.

    This class provides an event system for components to communicate with each other
    through events.
    """

    _instance = None

    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(EventSystem, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the event system."""
        if self._initialized:
            return

        self._listeners: Dict[EventType, List[EventListener]] = {}
        self._async_listeners: Dict[EventType, List[AsyncEventListener]] = {}
        self._all_listeners: List[EventListener] = []
        self._all_async_listeners: List[AsyncEventListener] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._initialized = True

        logger.info("Initialized EventSystem")

    def add_listener(
        self,
        event_type: Union[EventType, List[EventType]],
        listener: Union[EventListener, AsyncEventListener],
        is_async: bool = False,
    ) -> None:
        """
        Add a listener for an event type.

        Args:
            event_type: Type of event to listen for, or list of event types
            listener: Function to call when the event occurs
            is_async: Whether the listener is asynchronous
        """
        if isinstance(event_type, list):
            for et in event_type:
                self.add_listener(et, listener, is_async)
            return

        if event_type == EventType.CUSTOM:
            if is_async:
                self._all_async_listeners.append(listener)
            else:
                self._all_listeners.append(listener)
            logger.debug(f"Added listener for all events: {listener.__name__}")
            return

        if is_async:
            if event_type not in self._async_listeners:
                self._async_listeners[event_type] = []
            self._async_listeners[event_type].append(listener)
        else:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(listener)

        logger.debug(f"Added listener for {event_type}: {listener.__name__}")

    def remove_listener(
        self,
        event_type: Union[EventType, List[EventType]],
        listener: Union[EventListener, AsyncEventListener],
        is_async: bool = False,
    ) -> None:
        """
        Remove a listener for an event type.

        Args:
            event_type: Type of event to stop listening for, or list of event types
            listener: Function to remove
            is_async: Whether the listener is asynchronous
        """
        if isinstance(event_type, list):
            for et in event_type:
                self.remove_listener(et, listener, is_async)
            return

        if event_type == EventType.CUSTOM:
            if is_async:
                if listener in self._all_async_listeners:
                    self._all_async_listeners.remove(listener)
            else:
                if listener in self._all_listeners:
                    self._all_listeners.remove(listener)
            logger.debug(f"Removed listener for all events: {listener.__name__}")
            return

        if is_async:
            if event_type in self._async_listeners and listener in self._async_listeners[event_type]:
                self._async_listeners[event_type].remove(listener)
                logger.debug(f"Removed async listener for {event_type}: {listener.__name__}")
        else:
            if event_type in self._listeners and listener in self._listeners[event_type]:
                self._listeners[event_type].remove(listener)
                logger.debug(f"Removed listener for {event_type}: {listener.__name__}")

    def emit(self, event: Event) -> None:
        """
        Emit an event.

        Args:
            event: Event to emit
        """
        # Add the event to the queue
        self._event_queue.put_nowait(event)

        # Start processing if not already running
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_events())

        logger.debug(f"Emitted event: {event.type} from {event.source}")

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while not self._event_queue.empty():
            try:
                # Get the next event
                event = await self._event_queue.get()

                # Call synchronous listeners
                self._call_sync_listeners(event)

                # Call asynchronous listeners
                await self._call_async_listeners(event)

                # Mark the event as processed
                self._event_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Event processing cancelled")
                break
            except Exception as e:
                logger.exception(f"Error processing event: {e}")

    def _call_sync_listeners(self, event: Event) -> None:
        """
        Call synchronous listeners for an event.

        Args:
            event: Event to process
        """
        # Call listeners for this event type
        for listener in self._listeners.get(event.type, []):
            try:
                listener(event)
            except Exception as e:
                logger.exception(f"Error in listener {listener.__name__} for {event.type}: {e}")

        # Call listeners for all events
        for listener in self._all_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.exception(f"Error in all-events listener {listener.__name__}: {e}")

    async def _call_async_listeners(self, event: Event) -> None:
        """
        Call asynchronous listeners for an event.

        Args:
            event: Event to process
        """
        # Call async listeners for this event type
        for listener in self._async_listeners.get(event.type, []):
            try:
                await listener(event)
            except Exception as e:
                logger.exception(f"Error in async listener {listener.__name__} for {event.type}: {e}")

        # Call async listeners for all events
        for listener in self._all_async_listeners:
            try:
                await listener(event)
            except Exception as e:
                logger.exception(f"Error in all-events async listener {listener.__name__}: {e}")

    def clear_listeners(self) -> None:
        """Clear all listeners."""
        self._listeners.clear()
        self._async_listeners.clear()
        self._all_listeners.clear()
        self._all_async_listeners.clear()
        logger.info("Cleared all event listeners")

    def get_listener_count(self, event_type: Optional[EventType] = None) -> int:
        """
        Get the number of listeners for an event type.

        Args:
            event_type: Type of event to count listeners for, or None for all

        Returns:
            int: Number of listeners
        """
        if event_type is None:
            # Count all listeners
            count = len(self._all_listeners) + len(self._all_async_listeners)
            for listeners in self._listeners.values():
                count += len(listeners)
            for listeners in self._async_listeners.values():
                count += len(listeners)
            return count

        # Count listeners for this event type
        count = 0
        if event_type in self._listeners:
            count += len(self._listeners[event_type])
        if event_type in self._async_listeners:
            count += len(self._async_listeners[event_type])
        return count
