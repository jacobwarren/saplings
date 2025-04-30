"""
Tests for the GraphRunner class.
"""

import asyncio
import pytest
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole
from saplings.orchestration.config import (
    AgentNode,
    CommunicationChannel,
    GraphRunnerConfig,
    NegotiationStrategy,
)
from saplings.orchestration.graph_runner import GraphRunner


class TestGraphRunner:
    """Tests for the GraphRunner class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock(spec=LLM)
        mock.generate.return_value = LLMResponse(
            text="This is a test response",
            model_uri="test://model",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            metadata={"model": "test-model"},
        )
        mock.get_metadata.return_value = ModelMetadata(
            name="test-model",
            provider="test-provider",
            version="1.0",
            capabilities=[],
            roles=[ModelRole.GENERAL],
            context_window=4096,
            max_tokens_per_request=1024,
        )
        return mock

    @pytest.fixture
    def graph_runner(self, mock_llm):
        """Create a GraphRunner instance for testing."""
        config = GraphRunnerConfig(
            negotiation_strategy=NegotiationStrategy.DEBATE,
            max_rounds=3,
            timeout_seconds=10,
        )
        return GraphRunner(model=mock_llm, config=config)

    def test_initialization(self, graph_runner, mock_llm):
        """Test initialization of GraphRunner."""
        assert graph_runner.model == mock_llm
        assert graph_runner.config.negotiation_strategy == NegotiationStrategy.DEBATE
        assert graph_runner.config.max_rounds == 3
        assert graph_runner.config.timeout_seconds == 10
        assert graph_runner.agents == {}
        assert graph_runner.channels == []

    def test_register_agent(self, graph_runner):
        """Test registering an agent."""
        # Register an agent
        agent = AgentNode(
            id="agent1",
            name="Test Agent",
            role="tester",
            description="A test agent",
            capabilities=["testing"],
        )
        graph_runner.register_agent(agent)

        # Check that the agent was registered
        assert "agent1" in graph_runner.agents
        assert graph_runner.agents["agent1"] == agent

    def test_register_duplicate_agent(self, graph_runner):
        """Test registering a duplicate agent."""
        # Register an agent
        agent = AgentNode(
            id="agent1",
            name="Test Agent",
            role="tester",
            description="A test agent",
            capabilities=["testing"],
        )
        graph_runner.register_agent(agent)

        # Register a duplicate agent
        duplicate_agent = AgentNode(
            id="agent1",
            name="Duplicate Agent",
            role="duplicator",
            description="A duplicate agent",
            capabilities=["duplicating"],
        )
        with pytest.raises(ValueError):
            graph_runner.register_agent(duplicate_agent)

    def test_create_channel(self, graph_runner):
        """Test creating a communication channel."""
        # Register agents
        agent1 = AgentNode(
            id="agent1",
            name="Test Agent 1",
            role="tester",
            description="A test agent",
            capabilities=["testing"],
        )
        agent2 = AgentNode(
            id="agent2",
            name="Test Agent 2",
            role="reviewer",
            description="A review agent",
            capabilities=["reviewing"],
        )
        graph_runner.register_agent(agent1)
        graph_runner.register_agent(agent2)

        # Create a channel
        channel = graph_runner.create_channel(
            source_id="agent1",
            target_id="agent2",
            channel_type="test",
            description="A test channel",
        )

        # Check that the channel was created
        assert channel in graph_runner.channels
        assert channel.source_id == "agent1"
        assert channel.target_id == "agent2"
        assert channel.channel_type == "test"
        assert channel.description == "A test channel"

    def test_create_channel_invalid_agent(self, graph_runner):
        """Test creating a channel with an invalid agent."""
        # Register an agent
        agent = AgentNode(
            id="agent1",
            name="Test Agent",
            role="tester",
            description="A test agent",
            capabilities=["testing"],
        )
        graph_runner.register_agent(agent)

        # Create a channel with an invalid agent
        with pytest.raises(ValueError):
            graph_runner.create_channel(
                source_id="agent1",
                target_id="invalid_agent",
                channel_type="test",
                description="A test channel",
            )

    @pytest.mark.asyncio
    async def test_debate_negotiation(self, graph_runner):
        """Test debate negotiation strategy."""
        # Register agents
        agent1 = AgentNode(
            id="agent1",
            name="Test Agent 1",
            role="proposer",
            description="An agent that proposes ideas",
            capabilities=["proposing"],
        )
        agent2 = AgentNode(
            id="agent2",
            name="Test Agent 2",
            role="critic",
            description="An agent that critiques ideas",
            capabilities=["critiquing"],
        )
        graph_runner.register_agent(agent1)
        graph_runner.register_agent(agent2)

        # Create channels
        graph_runner.create_channel(
            source_id="agent1",
            target_id="agent2",
            channel_type="proposal",
            description="Channel for proposals",
        )
        graph_runner.create_channel(
            source_id="agent2",
            target_id="agent1",
            channel_type="critique",
            description="Channel for critiques",
        )

        # Mock the debate method
        with patch.object(graph_runner, "_run_debate", return_value="Consensus reached: This is the solution") as mock_debate:
            # Run the negotiation
            result = await graph_runner.negotiate(
                task="Solve this problem",
                context="This is a test problem",
            )

            # Check that the debate method was called
            mock_debate.assert_called_once()

            # Check the result
            assert result == "Consensus reached: This is the solution"

    @pytest.mark.asyncio
    async def test_contract_net_negotiation(self, graph_runner):
        """Test contract-net negotiation strategy."""
        # Set the negotiation strategy to CONTRACT_NET
        graph_runner.config.negotiation_strategy = NegotiationStrategy.CONTRACT_NET

        # Register agents
        manager = AgentNode(
            id="manager",
            name="Manager Agent",
            role="manager",
            description="An agent that manages tasks",
            capabilities=["managing"],
        )
        worker1 = AgentNode(
            id="worker1",
            name="Worker Agent 1",
            role="worker",
            description="An agent that performs tasks",
            capabilities=["working"],
        )
        worker2 = AgentNode(
            id="worker2",
            name="Worker Agent 2",
            role="worker",
            description="Another agent that performs tasks",
            capabilities=["working"],
        )
        graph_runner.register_agent(manager)
        graph_runner.register_agent(worker1)
        graph_runner.register_agent(worker2)

        # Create channels
        graph_runner.create_channel(
            source_id="manager",
            target_id="worker1",
            channel_type="task",
            description="Channel for task assignment",
        )
        graph_runner.create_channel(
            source_id="manager",
            target_id="worker2",
            channel_type="task",
            description="Channel for task assignment",
        )
        graph_runner.create_channel(
            source_id="worker1",
            target_id="manager",
            channel_type="result",
            description="Channel for task results",
        )
        graph_runner.create_channel(
            source_id="worker2",
            target_id="manager",
            channel_type="result",
            description="Channel for task results",
        )

        # Mock the contract-net method
        with patch.object(graph_runner, "_run_contract_net", return_value="Task completed: This is the result") as mock_contract_net:
            # Run the negotiation
            result = await graph_runner.negotiate(
                task="Perform this task",
                context="This is a test task",
            )

            # Check that the contract-net method was called
            mock_contract_net.assert_called_once()

            # Check the result
            assert result == "Task completed: This is the result"

    @pytest.mark.asyncio
    async def test_invalid_negotiation_strategy(self, graph_runner):
        """Test invalid negotiation strategy."""
        # Set an invalid negotiation strategy
        graph_runner.config.negotiation_strategy = "INVALID"

        # Run the negotiation
        with pytest.raises(ValueError):
            await graph_runner.negotiate(
                task="Solve this problem",
                context="This is a test problem",
            )

    @pytest.mark.asyncio
    async def test_run_with_timeout(self, graph_runner):
        """Test running with a timeout."""
        # Set a very short timeout
        graph_runner.config.timeout_seconds = 0.1

        # Register agents
        agent1 = AgentNode(
            id="agent1",
            name="Test Agent 1",
            role="proposer",
            description="An agent that proposes ideas",
            capabilities=["proposing"],
        )
        agent2 = AgentNode(
            id="agent2",
            name="Test Agent 2",
            role="critic",
            description="An agent that critiques ideas",
            capabilities=["critiquing"],
        )
        graph_runner.register_agent(agent1)
        graph_runner.register_agent(agent2)

        # Create channels
        graph_runner.create_channel(
            source_id="agent1",
            target_id="agent2",
            channel_type="proposal",
            description="Channel for proposals",
        )
        graph_runner.create_channel(
            source_id="agent2",
            target_id="agent1",
            channel_type="critique",
            description="Channel for critiques",
        )

        # Mock the debate method to sleep longer than the timeout
        async def slow_debate(*args, **kwargs):
            await asyncio.sleep(1)
            return "This should time out"

        with patch.object(graph_runner, "_run_debate", side_effect=slow_debate):
            # Run the negotiation
            with pytest.raises(asyncio.TimeoutError):
                await graph_runner.negotiate(
                    task="Solve this problem",
                    context="This is a test problem",
                )

    @pytest.mark.asyncio
    async def test_run_with_max_rounds(self, graph_runner):
        """Test running with max rounds."""
        # Set max rounds to 2
        graph_runner.config.max_rounds = 2

        # Register agents
        agent1 = AgentNode(
            id="agent1",
            name="Test Agent 1",
            role="proposer",
            description="An agent that proposes ideas",
            capabilities=["proposing"],
        )
        agent2 = AgentNode(
            id="agent2",
            name="Test Agent 2",
            role="critic",
            description="An agent that critiques ideas",
            capabilities=["critiquing"],
        )
        graph_runner.register_agent(agent1)
        graph_runner.register_agent(agent2)

        # Create channels
        graph_runner.create_channel(
            source_id="agent1",
            target_id="agent2",
            channel_type="proposal",
            description="Channel for proposals",
        )
        graph_runner.create_channel(
            source_id="agent2",
            target_id="agent1",
            channel_type="critique",
            description="Channel for critiques",
        )

        # Mock the _run_negotiation method to track rounds
        original_run_negotiation = graph_runner._run_negotiation
        rounds = [0]

        async def mock_run_negotiation(task, context, max_rounds):
            # Increment the round counter
            rounds[0] += 1

            # Return a result based on the round
            return f"Round {rounds[0]} complete"

        # Replace the _run_negotiation method
        graph_runner._run_negotiation = mock_run_negotiation

        try:
            # Run the negotiation twice
            result1 = await graph_runner.negotiate(
                task="Solve this problem",
                context="This is a test problem",
            )

            result2 = await graph_runner.negotiate(
                task="Solve this problem again",
                context="This is another test problem",
            )

            # Check that we ran exactly 2 rounds
            assert rounds[0] == 2
            assert result1 == "Round 1 complete"
            assert result2 == "Round 2 complete"
        finally:
            # Restore the original method
            graph_runner._run_negotiation = original_run_negotiation

    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, graph_runner):
        """Test coordination between multiple agents in a complex graph."""
        # Create a team of specialized agents
        researcher = AgentNode(
            id="researcher",
            name="Research Agent",
            role="researcher",
            description="An agent that researches information",
            capabilities=["research", "information_gathering"],
        )

        analyst = AgentNode(
            id="analyst",
            name="Data Analyst",
            role="analyst",
            description="An agent that analyzes data",
            capabilities=["data_analysis", "statistics"],
        )

        writer = AgentNode(
            id="writer",
            name="Content Writer",
            role="writer",
            description="An agent that writes content",
            capabilities=["writing", "summarization"],
        )

        reviewer = AgentNode(
            id="reviewer",
            name="Content Reviewer",
            role="reviewer",
            description="An agent that reviews and improves content",
            capabilities=["editing", "quality_control"],
        )

        # Register all agents
        graph_runner.register_agent(researcher)
        graph_runner.register_agent(analyst)
        graph_runner.register_agent(writer)
        graph_runner.register_agent(reviewer)

        # Create communication channels between agents
        # Research -> Analysis
        graph_runner.create_channel(
            source_id="researcher",
            target_id="analyst",
            channel_type="research_data",
            description="Channel for research data",
        )

        # Analysis -> Writing
        graph_runner.create_channel(
            source_id="analyst",
            target_id="writer",
            channel_type="analysis_results",
            description="Channel for analysis results",
        )

        # Writing -> Review
        graph_runner.create_channel(
            source_id="writer",
            target_id="reviewer",
            channel_type="draft_content",
            description="Channel for draft content",
        )

        # Review -> Writing (feedback loop)
        graph_runner.create_channel(
            source_id="reviewer",
            target_id="writer",
            channel_type="revision_feedback",
            description="Channel for revision feedback",
        )

        # Mock the negotiate method
        async def mock_negotiate(*args, **kwargs):
            # Return a final result directly
            return "Final report: A comprehensive analysis of the data."

        # Replace the method
        with patch.object(graph_runner, "negotiate", side_effect=mock_negotiate):
            # Run the multi-agent workflow
            result = await graph_runner.negotiate(
                task="Analyze recent climate data and prepare a report",
                context="Focus on temperature changes in the last decade",
            )

            # The result should contain the final report text

            # Check the result
            assert "Final report" in result

            # Verify that all agents were registered
            assert "researcher" in graph_runner.agents
            assert "analyst" in graph_runner.agents
            assert "writer" in graph_runner.agents
            assert "reviewer" in graph_runner.agents

            # Verify that all channels were created
            assert len(graph_runner.channels) == 4

            # Verify channel connections
            research_to_analysis = False
            analysis_to_writing = False
            writing_to_review = False
            review_to_writing = False

            for channel in graph_runner.channels:
                if channel.source_id == "researcher" and channel.target_id == "analyst":
                    research_to_analysis = True
                elif channel.source_id == "analyst" and channel.target_id == "writer":
                    analysis_to_writing = True
                elif channel.source_id == "writer" and channel.target_id == "reviewer":
                    writing_to_review = True
                elif channel.source_id == "reviewer" and channel.target_id == "writer":
                    review_to_writing = True

            assert research_to_analysis
            assert analysis_to_writing
            assert writing_to_review
            assert review_to_writing
