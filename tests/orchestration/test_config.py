"""
Tests for the orchestration configuration module.
"""

import pytest
from pydantic import ValidationError

from saplings.orchestration.config import (
    AgentNode,
    CommunicationChannel,
    GraphRunnerConfig,
    NegotiationStrategy,
)


class TestOrchestrationConfig:
    """Tests for the orchestration configuration classes."""

    def test_agent_node(self):
        """Test AgentNode configuration."""
        # Test valid configuration
        agent = AgentNode(
            id="agent1",
            name="Test Agent",
            role="tester",
            description="A test agent",
            capabilities=["testing"],
        )
        assert agent.id == "agent1"
        assert agent.name == "Test Agent"
        assert agent.role == "tester"
        assert agent.description == "A test agent"
        assert agent.capabilities == ["testing"]
        assert agent.metadata == {}

        # Test with metadata
        agent = AgentNode(
            id="agent1",
            name="Test Agent",
            role="tester",
            description="A test agent",
            capabilities=["testing"],
            metadata={"key": "value"},
        )
        assert agent.metadata == {"key": "value"}

        # Test invalid ID (empty string)
        with pytest.raises(ValidationError):
            AgentNode(
                id="",
                name="Test Agent",
                role="tester",
                description="A test agent",
                capabilities=["testing"],
            )

    def test_communication_channel(self):
        """Test CommunicationChannel configuration."""
        # Test valid configuration
        channel = CommunicationChannel(
            source_id="agent1",
            target_id="agent2",
            channel_type="test",
            description="A test channel",
        )
        assert channel.source_id == "agent1"
        assert channel.target_id == "agent2"
        assert channel.channel_type == "test"
        assert channel.description == "A test channel"
        assert channel.metadata == {}

        # Test with metadata
        channel = CommunicationChannel(
            source_id="agent1",
            target_id="agent2",
            channel_type="test",
            description="A test channel",
            metadata={"key": "value"},
        )
        assert channel.metadata == {"key": "value"}

        # Test invalid source_id (empty string)
        with pytest.raises(ValidationError):
            CommunicationChannel(
                source_id="",
                target_id="agent2",
                channel_type="test",
                description="A test channel",
            )

        # Test invalid target_id (empty string)
        with pytest.raises(ValidationError):
            CommunicationChannel(
                source_id="agent1",
                target_id="",
                channel_type="test",
                description="A test channel",
            )

    def test_graph_runner_config(self):
        """Test GraphRunnerConfig configuration."""
        # Test default configuration
        config = GraphRunnerConfig()
        assert config.negotiation_strategy == NegotiationStrategy.DEBATE
        assert config.max_rounds == 5
        assert config.timeout_seconds == 60
        assert config.consensus_threshold == 0.8
        assert config.logging_enabled is True
        assert config.metadata == {}

        # Test custom configuration
        config = GraphRunnerConfig(
            negotiation_strategy=NegotiationStrategy.CONTRACT_NET,
            max_rounds=10,
            timeout_seconds=120,
            consensus_threshold=0.9,
            logging_enabled=False,
            metadata={"key": "value"},
        )
        assert config.negotiation_strategy == NegotiationStrategy.CONTRACT_NET
        assert config.max_rounds == 10
        assert config.timeout_seconds == 120
        assert config.consensus_threshold == 0.9
        assert config.logging_enabled is False
        assert config.metadata == {"key": "value"}

        # Test invalid consensus_threshold (> 1.0)
        with pytest.raises(ValidationError):
            GraphRunnerConfig(consensus_threshold=1.1)

        # Test invalid consensus_threshold (< 0.0)
        with pytest.raises(ValidationError):
            GraphRunnerConfig(consensus_threshold=-0.1)

        # Test invalid max_rounds (< 1)
        with pytest.raises(ValidationError):
            GraphRunnerConfig(max_rounds=0)

        # Test invalid timeout_seconds (< 1)
        with pytest.raises(ValidationError):
            GraphRunnerConfig(timeout_seconds=0)
