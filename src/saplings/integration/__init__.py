from __future__ import annotations

"""
Integration module for Saplings.

This module re-exports the public API from saplings.api.integration.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides integration capabilities for Saplings, including:
- Hot-loading mechanism for tools
- Tool lifecycle management
- Integration with executor and planner
- Event system for cross-component communication
- Secure hot-loading with sandboxing
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.integration.

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


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        from saplings.api.integration import (
            Event,
            EventListener,
            EventSystem,
            EventType,
            HotLoader,
            HotLoaderConfig,
            IntegrationManager,
            SecureHotLoader,
            SecureHotLoaderConfig,
            ToolLifecycleManager,
            create_secure_hot_loader,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "Event": Event,
            "EventListener": EventListener,
            "EventSystem": EventSystem,
            "EventType": EventType,
            "HotLoader": HotLoader,
            "HotLoaderConfig": HotLoaderConfig,
            "IntegrationManager": IntegrationManager,
            "SecureHotLoader": SecureHotLoader,
            "SecureHotLoaderConfig": SecureHotLoaderConfig,
            "ToolLifecycleManager": ToolLifecycleManager,
            "create_secure_hot_loader": create_secure_hot_loader,
        }

        # Return the requested attribute
        return globals_dict.get(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
