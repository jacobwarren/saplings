"""
Configuration module for the orchestration system.

This module defines the configuration classes for the orchestration system.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NegotiationStrategy(str, Enum):
    """Strategy for agent negotiation."""

    DEBATE = "debate"  # Agents debate to reach consensus
    CONTRACT_NET = "contract_net"  # Contract Net Protocol for task delegation


class AgentNode(BaseModel):
    """Configuration for an agent node in the graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Human-readable name for the agent")
    role: str = Field(..., description="Role of the agent (e.g., 'planner', 'executor')")
    description: str = Field(..., description="Description of the agent's purpose")
    capabilities: List[str] = Field(default_factory=list, description="List of agent capabilities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Fields for model specification
    provider: Optional[str] = Field(
        None, description="Model provider (e.g., 'vllm', 'openai', 'anthropic')"
    )
    model: Optional[str] = Field(None, description="Model name")
    model_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model parameters"
    )

    # Fields for component integration
    memory_store: Optional[Any] = Field(default=None, description="Memory store for this agent")
    retriever: Optional[Any] = Field(default=None, description="Retriever for this agent")
    enable_gasa: bool = Field(default=False, description="Whether to enable GASA for this agent")
    gasa_config: Optional[Dict[str, Any]] = Field(
        default=None, description="GASA configuration for this agent"
    )

    # Field for agent composition
    agent: Optional[Any] = Field(default=None, description="Base Agent instance for this node")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate the agent ID."""
        if not v:
            raise ValueError("Agent ID cannot be empty")
        return v


class CommunicationChannel(BaseModel):
    """Configuration for a communication channel between agents."""

    source_id: str = Field(..., description="ID of the source agent")
    target_id: str = Field(..., description="ID of the target agent")
    channel_type: str = Field(..., description="Type of channel (e.g., 'task', 'result')")
    description: str = Field(..., description="Description of the channel's purpose")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("source_id", "target_id")
    @classmethod
    def validate_agent_ids(cls, v: str) -> str:
        """Validate the agent IDs."""
        if not v:
            raise ValueError("Agent ID cannot be empty")
        return v


class GraphRunnerConfig(BaseModel):
    """Configuration for the GraphRunner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    negotiation_strategy: NegotiationStrategy = Field(
        default=NegotiationStrategy.DEBATE,
        description="Strategy for agent negotiation",
    )
    max_rounds: int = Field(
        default=5,
        description="Maximum number of negotiation rounds",
        ge=1,
    )
    timeout_seconds: int = Field(
        default=60,
        description="Timeout for negotiation in seconds",
        ge=1,
    )
    consensus_threshold: float = Field(
        default=0.8,
        description="Threshold for consensus (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    logging_enabled: bool = Field(
        default=True,
        description="Whether to log agent interactions",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    # Memory and retrieval
    memory_store: Optional[Any] = Field(
        default=None,
        description="Shared memory store for all agents",
    )

    # Monitoring and tracing
    enable_monitoring: bool = Field(
        default=False,
        description="Whether to enable monitoring and tracing",
    )
    trace_manager: Optional[Any] = Field(
        default=None,
        description="Trace manager for monitoring agent interactions",
    )
    blame_graph: Optional[Any] = Field(
        default=None,
        description="Blame graph for identifying performance bottlenecks",
    )

    # Validation and judging
    enable_validation: bool = Field(
        default=False,
        description="Whether to enable validation of agent outputs",
    )
    judge: Optional[Any] = Field(
        default=None,
        description="Judge agent for validating outputs",
    )

    # Self-healing
    enable_self_healing: bool = Field(
        default=False,
        description="Whether to enable self-healing capabilities",
    )
    success_pair_collector: Optional[Any] = Field(
        default=None,
        description="Collector for successful error-fix pairs",
    )


# Rebuild the models to resolve circular references
AgentNode.model_rebuild()
GraphRunnerConfig.model_rebuild()
