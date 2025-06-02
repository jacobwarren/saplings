from __future__ import annotations

"""
Integration tests for service interactions.

These tests verify that services can interact with each other correctly.
"""


from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from saplings.agent_config import AgentConfig
from saplings.api.core.interfaces import (
    IExecutionService,
    IMemoryManager,
    IModelInitializationService,
    IPlannerService,
    IRetrievalService,
    IToolService,
    IValidatorService,
)
from saplings.container_config import configure_container
from saplings.di import reset_container


@pytest.fixture()
def test_config() -> None:
    """Create a test configuration."""
    test_dir = Path(__file__).parent / "test_output"
    test_dir.mkdir(parents=True, exist_ok=True)

    return AgentConfig(
        provider="test",
        model_name="test-model",
        memory_path=str(test_dir / "memory"),
        output_dir=str(test_dir / "output"),
        enable_monitoring=False,
        enable_self_healing=False,
        enable_tool_factory=False,
    )


@pytest.fixture()
def test_container(test_config) -> None:
    """Create a test container with test configuration."""
    # Reset container before test
    reset_container()

    # Configure container
    container = configure_container(test_config)

    # Yield container for test
    yield container

    # Reset container after test
    reset_container()

    # Clean up test directories
    import shutil
    from pathlib import Path

    memory_path = Path(test_config.memory_path)
    if memory_path.parent.exists():
        shutil.rmtree(memory_path.parent)


class TestServiceInteractions:
    """Test interactions between services."""

    def test_service_resolution(self, test_container) -> None:
        """Test that all services can be resolved from the container."""
        # Resolve all services
        model_init_service = test_container.resolve(IModelInitializationService)
        memory_manager = test_container.resolve(IMemoryManager)
        retrieval_service = test_container.resolve(IRetrievalService)
        planner_service = test_container.resolve(IPlannerService)
        execution_service = test_container.resolve(IExecutionService)
        validator_service = test_container.resolve(IValidatorService)
        tool_service = test_container.resolve(IToolService)

        # Verify services were resolved
        assert model_init_service is not None
        assert memory_manager is not None
        assert retrieval_service is not None
        assert planner_service is not None
        assert execution_service is not None
        assert validator_service is not None
        assert tool_service is not None

    def test_memory_retrieval_interaction(self, test_container) -> None:
        """Test interaction between memory manager and retrieval service."""
        # Get services
        memory_manager = test_container.resolve(IMemoryManager)
        retrieval_service = test_container.resolve(IRetrievalService)

        # Add documents to memory
        docs = []
        for i in range(5):
            doc = memory_manager.add_document(
                content=f"Test document {i} about artificial intelligence.",
                metadata={"source": f"doc{i}.txt"},
            )
            docs.append(doc)

        # Retrieve documents
        results = retrieval_service.retrieve("artificial intelligence", limit=3)

        # Verify results
        assert len(results) > 0
        assert all(isinstance(result[0], type(docs[0])) for result in results)
        assert all(isinstance(result[1], float) for result in results)

        # Verify content
        assert any("artificial intelligence" in result[0].content.lower() for result in results)

    @patch("saplings.core.model_adapter.LLM")
    async def test_model_execution_interaction(self, mock_llm, test_container) -> None:
        """Test interaction between model initialization service and execution service."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.text = "This is a test response."
        mock_llm.create.return_value.generate = MagicMock(return_value=mock_response)

        # Get services
        # We need to resolve the model initialization service to ensure it's initialized
        _ = test_container.resolve(IModelInitializationService)
        execution_service = test_container.resolve(IExecutionService)

        # Execute prompt
        response = await execution_service.execute(prompt="This is a test prompt.", documents=[])

        # Verify response
        assert response is not None
        assert response.text == "This is a test response."

    @patch("saplings.core.model_adapter.LLM")
    def test_memory_retrieval_execution_interaction(self, mock_llm, test_container) -> None:
        """Test interaction between memory, retrieval, and execution services."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.text = "This is a test response about artificial intelligence."
        mock_llm.create.return_value.generate = MagicMock(return_value=mock_response)

        # Get services
        memory_manager = test_container.resolve(IMemoryManager)
        retrieval_service = test_container.resolve(IRetrievalService)
        execution_service = test_container.resolve(IExecutionService)

        # Add documents to memory
        for i in range(5):
            memory_manager.add_document(
                content=f"Test document {i} about artificial intelligence.",
                metadata={"source": f"doc{i}.txt"},
            )

        # Retrieve documents
        results = retrieval_service.retrieve("artificial intelligence", limit=3)

        # Execute prompt with retrieved context
        response = execution_service.execute(
            prompt="Summarize the information about artificial intelligence.",
            context=[result[0].content for result in results],
        )

        # Verify response
        assert response is not None
        assert response.text == "This is a test response about artificial intelligence."
