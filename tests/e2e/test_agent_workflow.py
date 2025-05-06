from __future__ import annotations

"""
End-to-end tests for agent workflow.

These tests verify the complete agent workflow from user perspective.
They require API keys for external services and will be skipped if they are not available.
"""


import asyncio
import os

import pytest

from saplings.agent import Agent
from saplings.agent_config import AgentConfig


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available")
class TestAgentWorkflow:
    """Test complete agent workflow."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create test directory
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(self.test_dir, exist_ok=True)

        # Create agent
        self.agent = Agent(
            config=AgentConfig(
                provider="openai",
                model_name="gpt-3.5-turbo",
                memory_path=os.path.join(self.test_dir, "memory"),
                output_dir=os.path.join(self.test_dir, "output"),
                enable_gasa=True,
                enable_monitoring=True,
            )
        )

        # Add documents to memory
        for i in range(3):
            self.agent.memory_store.add_document(
                content=f"Document {i} about artificial intelligence and machine learning.",
                metadata={"source": f"doc{i}.txt"},
            )

    def teardown_method(self) -> None:
        """Clean up after test."""
        # Clean up test directories
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_agent_run(self) -> None:
        """Test agent run method."""
        # Run a query
        result = asyncio.run(self.agent.run("Summarize the documents about machine learning"))

        # Verify result
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

        # The result should mention machine learning
        assert "machine learning" in result.lower() or "artificial intelligence" in result.lower()

    def test_agent_run_with_tools(self) -> None:
        """Test agent run method with tools."""

        # Create a simple calculator tool
        def calculator(expression):
            return eval(expression)

        # Register tool
        self.agent.register_tool(
            name="calculator", description="Calculate mathematical expressions", function=calculator
        )

        # Run a query that requires the calculator
        result = asyncio.run(self.agent.run("What is 2 + 2?"))

        # Verify result
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

        # The result should mention 4
        assert "4" in result

    def test_agent_memory_integration(self) -> None:
        """Test agent memory integration."""
        # Add a document
        self.agent.memory_store.add_document(
            content="GPT-4 is a large language model developed by OpenAI.",
            metadata={"source": "gpt4.txt"},
        )

        # Run a query about the document
        result = asyncio.run(self.agent.run("What is GPT-4?"))

        # Verify result
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

        # The result should mention OpenAI
        assert "openai" in result.lower() or "language model" in result.lower()

    def test_agent_retrieval_integration(self) -> None:
        """Test agent retrieval integration."""
        # Add documents with specific content
        self.agent.memory_store.add_document(
            content="Python is a programming language created by Guido van Rossum.",
            metadata={"source": "python.txt"},
        )

        self.agent.memory_store.add_document(
            content="JavaScript is a programming language commonly used for web development.",
            metadata={"source": "javascript.txt"},
        )

        # Run a query about programming languages
        result = asyncio.run(self.agent.run("Compare Python and JavaScript"))

        # Verify result
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

        # The result should mention both languages
        assert "python" in result.lower()
        assert "javascript" in result.lower()

    @pytest.mark.skipif(True, reason="This test is slow and requires GASA to be fully implemented")
    def test_agent_gasa_integration(self) -> None:
        """Test agent GASA integration."""
        # Add documents with relationships
        doc1 = self.agent.memory_store.add_document(
            content="Neural networks are a type of machine learning model inspired by the human brain.",
            metadata={"source": "neural_networks.txt"},
        )

        doc2 = self.agent.memory_store.add_document(
            content="Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images.",
            metadata={"source": "cnn.txt"},
        )

        # Add relationship
        self.agent.dependency_graph.add_relationship(doc1.id, doc2.id, "specializes", 0.9)

        # Run a query that requires understanding the relationship
        result = asyncio.run(self.agent.run("Explain how CNNs relate to neural networks"))

        # Verify result
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

        # The result should mention both neural networks and CNNs
        assert "neural network" in result.lower()
        assert "cnn" in result.lower()

    def test_agent_monitoring_integration(self) -> None:
        """Test agent monitoring integration."""
        # Run a query
        result = asyncio.run(self.agent.run("What is artificial intelligence?"))

        # Verify result
        assert result

        # Check traces
        traces = self.agent.monitoring_service.get_traces()
        assert traces
        assert len(traces) > 0
