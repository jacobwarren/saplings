from __future__ import annotations

"""
Orchestration module for Saplings.

This module re-exports the public API from saplings.api.orchestration.
For application code, it is recommended to import directly from saplings.api
or the top-level saplings package.

This module provides multi-agent orchestration capabilities for Saplings, including:
- GraphRunner for agent coordination
- Debate and contract-net negotiation strategies
- Integration with executor and planner
"""

# We don't import anything directly here to avoid circular imports.
# The public API is defined in saplings.api.orchestration.

__all__ = [
    "AgentNode",
    "CommunicationChannel",
    "GraphRunner",
    "GraphRunnerConfig",
    "NegotiationStrategy",
    "OrchestrationConfig",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import attributes to avoid circular dependencies."""
    if name in __all__:
        from saplings.api.orchestration import (
            AgentNode,
            CommunicationChannel,
            GraphRunner,
            GraphRunnerConfig,
            NegotiationStrategy,
            OrchestrationConfig,
        )

        # Create a mapping of names to their values
        globals_dict = {
            "AgentNode": AgentNode,
            "CommunicationChannel": CommunicationChannel,
            "GraphRunner": GraphRunner,
            "GraphRunnerConfig": GraphRunnerConfig,
            "NegotiationStrategy": NegotiationStrategy,
            "OrchestrationConfig": OrchestrationConfig,
        }

        # Return the requested attribute
        return globals_dict[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
