from __future__ import annotations

"""
Integration module for Saplings.

This module provides integration capabilities for Saplings, including:
- Hot-loading mechanism for tools
- Tool lifecycle management
- Integration with executor and planner
- Event system for cross-component communication
- Secure hot-loading with sandboxing
"""


from saplings.integration.events import Event, EventListener, EventSystem, EventType
from saplings.integration.hot_loader import HotLoader, HotLoaderConfig, ToolLifecycleManager
from saplings.integration.integration_manager import IntegrationManager
from saplings.integration.secure_hot_loader import (
    SecureHotLoader,
    SecureHotLoaderConfig,
    create_secure_hot_loader,
)

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
