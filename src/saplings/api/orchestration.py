from __future__ import annotations

"""
Public API for orchestration.

This module provides the public API for multi-agent orchestration, including:
- GraphRunner for agent coordination
- Debate and contract-net negotiation strategies
- Integration with executor and planner
"""

from saplings.api.stability import beta
from saplings.orchestration._internal import (
    AgentNode as _AgentNode,
)
from saplings.orchestration._internal import (
    CommunicationChannel as _CommunicationChannel,
)
from saplings.orchestration._internal import (
    GraphRunner as _GraphRunner,
)
from saplings.orchestration._internal import (
    GraphRunnerConfig as _GraphRunnerConfig,
)
from saplings.orchestration._internal import (
    NegotiationStrategy as _NegotiationStrategy,
)
from saplings.orchestration._internal import (
    OrchestrationConfig as _OrchestrationConfig,
)


@beta
class AgentNode(_AgentNode):
    """Agent node in the orchestration graph."""


@beta
class CommunicationChannel(_CommunicationChannel):
    """Communication channel between agents."""


@beta
class GraphRunner(_GraphRunner):
    """
    Coordinator for multiple agents in a graph structure.

    This class provides functionality for:
    - Registering agents and defining relationships
    - Implementing debate and contract-net negotiation strategies
    - Executing multi-agent workflows
    - Tracking agent interactions
    """


@beta
class GraphRunnerConfig(_GraphRunnerConfig):
    """Configuration for the graph runner."""


# Use direct assignment for enums to avoid extending them
NegotiationStrategy = _NegotiationStrategy
# Add stability annotation
beta(NegotiationStrategy)


@beta
class OrchestrationConfig(_OrchestrationConfig):
    """Configuration for the orchestration system (alias for GraphRunnerConfig)."""


__all__ = [
    "AgentNode",
    "CommunicationChannel",
    "GraphRunner",
    "GraphRunnerConfig",
    "NegotiationStrategy",
    "OrchestrationConfig",
]
