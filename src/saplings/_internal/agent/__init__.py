from __future__ import annotations

"""
Internal agent module for Saplings.

This module provides the internal implementation of the Agent class and related components.
All components are organized in a way that avoids circular imports.
"""

import importlib
from typing import Any

# Define __all__ to control what is exported
__all__ = [
    "Agent",
    "AgentBuilder",
    "AgentConfig",
    "AgentFacade",
    "AgentFacadeBuilder",
]


# Use lazy imports to avoid circular dependencies
def __getattr__(name: str) -> Any:
    """Lazy import mechanism for modules to avoid circular imports."""
    if name in __all__:
        # Map names to their module paths
        module_map = {
            "Agent": "saplings._internal.agent.core",
            "AgentBuilder": "saplings._internal.agent.builder",
            "AgentConfig": "saplings._internal.agent.config",
            "AgentFacade": "saplings._internal.agent.facade",
            "AgentFacadeBuilder": "saplings._internal.agent.facade_builder",
        }

        if name in module_map:
            module = importlib.import_module(module_map[name])
            return getattr(module, name)

    raise AttributeError(f"module 'saplings._internal.agent' has no attribute '{name}'")
