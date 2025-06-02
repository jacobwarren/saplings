from __future__ import annotations

"""
Configuration module for the orchestration system.

This module defines the configuration classes for the orchestration system.
"""


from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NegotiationStrategy(str, Enum):
    """Strategy for agent negotiation."""

    DEBATE = "debate"  # Agents debate to reach consensus
    CONTRACT_NET = "contract_net"  # Contract Net Protocol for task delegation


class CommunicationChannel(BaseModel):
    """Communication channel between agents."""

    source: str = Field(
        ...,
        description="Source agent ID",
    )
    target: str = Field(
        ...,
        description="Target agent ID",
    )
    channel_type: str = Field(
        default="direct",
        description="Type of communication channel",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = ConfigDict(extra="allow")


class AgentNode(BaseModel):
    """Agent node in the orchestration graph."""

    id: str = Field(
        ...,
        description="Unique identifier for the agent",
    )
    name: str = Field(
        ...,
        description="Human-readable name for the agent",
    )
    description: str = Field(
        default="",
        description="Description of the agent's role",
    )
    agent_type: str = Field(
        default="default",
        description="Type of agent",
    )
    model_name: str | None = Field(
        default=None,
        description="Name of the model to use for this agent",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt for the agent",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="List of tool IDs available to this agent",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = ConfigDict(extra="allow")


class GraphRunnerConfig(BaseModel):
    """Configuration for the graph runner."""

    # Basic configuration
    id: str = Field(
        default="default",
        description="Unique identifier for this graph runner",
    )
    name: str = Field(
        default="Default Graph Runner",
        description="Human-readable name for this graph runner",
    )
    description: str = Field(
        default="",
        description="Description of this graph runner",
    )

    # Agent configuration
    agents: list[AgentNode] = Field(
        default_factory=list,
        description="List of agent nodes in the graph",
    )
    channels: list[CommunicationChannel] = Field(
        default_factory=list,
        description="List of communication channels between agents",
    )
    negotiation_strategy: NegotiationStrategy = Field(
        default=NegotiationStrategy.DEBATE,
        description="Strategy for agent negotiation",
    )

    # Model configuration
    default_model_name: str | None = Field(
        default=None,
        description="Default model name for agents",
    )
    default_system_prompt: str | None = Field(
        default=None,
        description="Default system prompt for agents",
    )

    # Execution configuration
    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations",
    )
    timeout_seconds: int = Field(
        default=300,
        description="Timeout in seconds",
    )
    parallel_execution: bool = Field(
        default=False,
        description="Whether to execute agents in parallel",
    )

    # Additional configuration
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    # Memory and retrieval
    memory_store: Any | None = Field(
        default=None,
        description="Shared memory store for all agents",
    )

    # Monitoring and tracing
    enable_monitoring: bool = Field(
        default=False,
        description="Whether to enable monitoring and tracing",
    )
    trace_manager: Any | None = Field(
        default=None,
        description="Trace manager for monitoring agent interactions",
    )
    blame_graph: Any | None = Field(
        default=None,
        description="Blame graph for identifying performance bottlenecks",
    )

    # Validation and judging
    enable_validation: bool = Field(
        default=False,
        description="Whether to enable validation of agent outputs",
    )
    judge: Any | None = Field(
        default=None,
        description="Judge agent for validating outputs",
    )

    # Self-healing
    enable_self_healing: bool = Field(
        default=False,
        description="Whether to enable self-healing capabilities",
    )
    success_pair_collector: Any | None = Field(
        default=None,
        description="Collector for successful error-fix pairs",
    )


# Define OrchestrationConfig as an alias for GraphRunnerConfig for backward compatibility
OrchestrationConfig = GraphRunnerConfig


# Rebuild the models to resolve circular references
AgentNode.model_rebuild()
GraphRunnerConfig.model_rebuild()
