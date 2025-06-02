from __future__ import annotations

"""
Agent module for Saplings.

This module re-exports the public API from saplings.api.agent.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.
"""

# Import from the public API
from saplings.api.agent import (
    Agent,
    AgentBuilder,
    AgentConfig,
    AgentFacade,
    AgentFacadeBuilder,
)

__all__ = [
    "Agent",
    "AgentBuilder",
    "AgentConfig",
    "AgentFacade",
    "AgentFacadeBuilder",
]
