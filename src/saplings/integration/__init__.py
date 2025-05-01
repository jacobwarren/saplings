"""
Integration module for Saplings.

This module provides integration capabilities for Saplings, including:
- Hot-loading mechanism for tools
- Tool lifecycle management
- Integration with executor and planner
- Event system for cross-component communication
"""

from saplings.integration.events import Event, EventListener, EventSystem, EventType
from saplings.integration.hot_loader import HotLoader, HotLoaderConfig, ToolLifecycleManager
from saplings.integration.integration_manager import IntegrationManager

__all__ = [
    "HotLoader",
    "HotLoaderConfig",
    "ToolLifecycleManager",
    "IntegrationManager",
    "EventSystem",
    "EventType",
    "Event",
    "EventListener",
]
