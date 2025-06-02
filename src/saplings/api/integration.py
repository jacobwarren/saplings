from __future__ import annotations

"""
Integration API module for Saplings.

This module provides the public API for integration capabilities, including:
- Hot-loading mechanism for tools
- Tool lifecycle management
- Integration with executor and planner
- Event system for cross-component communication
- Secure hot-loading with sandboxing
"""

from saplings.api.stability import beta, stable

# Import directly from internal modules
from saplings.integration._internal.service.events import (
    Event as _Event,
)
from saplings.integration._internal.service.events import (
    EventListener as _EventListener,
)
from saplings.integration._internal.service.events import (
    EventSystem as _EventSystem,
)
from saplings.integration._internal.service.events import (
    EventType as _EventType,
)
from saplings.integration._internal.service.hot_loader import (
    HotLoader as _HotLoader,
)
from saplings.integration._internal.service.hot_loader import (
    HotLoaderConfig as _HotLoaderConfig,
)
from saplings.integration._internal.service.integration_manager import (
    IntegrationManager as _IntegrationManager,
)
from saplings.integration._internal.service.integration_manager import (
    ToolLifecycleManager as _ToolLifecycleManager,
)
from saplings.integration._internal.service.secure_hot_loader import (
    SecureHotLoader as _SecureHotLoader,
)
from saplings.integration._internal.service.secure_hot_loader import (
    SecureHotLoaderConfig as _SecureHotLoaderConfig,
)
from saplings.integration._internal.service.secure_hot_loader import (
    create_secure_hot_loader as _create_secure_hot_loader,
)

# Re-export the EventType enum
EventType = _EventType


@stable
class Event(_Event):
    """
    Event data structure.

    This class represents an event in the Saplings event system, including
    the event type, source, data, and timestamp.
    """


@stable
class EventSystem(_EventSystem):
    """
    Event system for cross-component communication.

    This class provides an event system for components to communicate with each other
    through events.
    """


@stable
class HotLoaderConfig(_HotLoaderConfig):
    """
    Configuration for the hot-loading system.

    This class provides configuration options for the hot-loading system, including
    watch directories, auto-reload settings, and callbacks.
    """


@stable
class ToolLifecycleManager(_ToolLifecycleManager):
    """
    Manager for tool lifecycle.

    This class manages the lifecycle of tools, including initialization,
    cleanup, and state management.
    """


@stable
class HotLoader(_HotLoader):
    """
    Hot-loading system for tools.

    This class provides the hot-loading mechanism for tools, allowing them to be
    added, updated, or removed without restarting the system.
    """


@stable
class IntegrationManager(_IntegrationManager):
    """
    Manager for integration components.

    This class manages the integration of various components in the Saplings
    framework, including the event system, hot-loading system, and more.
    """


@beta
class SecureHotLoaderConfig(_SecureHotLoaderConfig):
    """
    Configuration for the secure hot-loading system.

    This class provides configuration options for the secure hot-loading system,
    including sandboxing options, security settings, and more.
    """


@beta
class SecureHotLoader(_SecureHotLoader):
    """
    Secure hot-loading system for tools.

    This class extends the standard HotLoader with additional security measures,
    particularly sandboxing for all dynamically loaded code.
    """


# Type alias for event listeners
EventListener = _EventListener


@beta
def create_secure_hot_loader(config=None):
    """
    Create a secure hot loader.

    This function creates a secure hot loader with the given configuration.

    Args:
    ----
        config: Configuration for the secure hot loader

    Returns:
    -------
        SecureHotLoader: Secure hot loader instance

    """
    return _create_secure_hot_loader(config)


__all__ = [
    "Event",
    "EventListener",
    "EventSystem",
    "EventType",
    "HotLoader",
    "HotLoaderConfig",
    "IntegrationManager",
    "SecureHotLoader",
    "SecureHotLoaderConfig",
    "ToolLifecycleManager",
    "create_secure_hot_loader",
]
