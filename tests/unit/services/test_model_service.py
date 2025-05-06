from __future__ import annotations

"""
Unit tests for the model service.
"""


from unittest.mock import MagicMock, patch

import pytest

from saplings.core.interfaces import IModelService
from saplings.core.model_adapter import LLM, LLMResponse
from saplings.services.model_service import ModelService


class TestModelService:
    EXPECTED_COUNT_1 = 2

    """Test the model service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create mock model adapter
        self.mock_adapter = MagicMock(spec=LLM)

        # Create a mock response
        self.mock_response = LLMResponse(
            text="Test response",
            provider="test",
            model_name="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            function_call=None,
            tool_calls=None,
        )

        # Create a mock for the generate method
        self.mock_generate = MagicMock()

        # Make it return a coroutine that returns the mock response
        async def _mock_generate(*args: object, **kwargs: object) -> LLMResponse:
            self.mock_generate(*args, **kwargs)
            return self.mock_response

        self.mock_adapter.generate = _mock_generate
        self.mock_adapter.generate_stream = MagicMock()

        # Patch the LLM.create method
        self.patcher = patch(
            "saplings.core.model_adapter.LLM.create", return_value=self.mock_adapter
        )
        self.mock_create = self.patcher.start()

        # Create model service
        self.service = ModelService(provider="test", model_name="test-model")

    def teardown_method(self) -> None:
        """Clean up after test."""
        self.patcher.stop()

    def test_initialization(self) -> None:
        """Test model service initialization."""
        assert self.service.provider == "test"
        assert self.service.model_name == "test-model"
        assert self.service.model is self.mock_adapter

    @pytest.mark.asyncio()
    async def test_get_model(self) -> None:
        """Test get_model method."""
        # Get model
        model = await self.service.get_model()

        # Verify model
        assert model is self.mock_adapter

        # Verify LLM.create was called
        self.mock_create.assert_called_once_with(
            provider="test",
            model_name="test-model",
        )

    @pytest.mark.asyncio()
    async def test_generate(self) -> None:
        """Test generate method."""
        # Generate text
        response = await self.service.generate(prompt="Test prompt")

        # Verify response
        assert response.text == "Test response"
        assert response.provider == "test"
        assert response.model_name == "test-model"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5
        assert response.usage["total_tokens"] == 15

        # Verify adapter was called
        self.mock_generate.assert_called_once()

    @pytest.mark.asyncio()
    async def test_generate_with_parameters(self) -> None:
        """Test generate method with parameters."""
        # Generate text with parameters
        response = await self.service.generate(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100,
        )

        # Verify response
        assert response.text == "Test response"

        # Verify adapter was called with parameters
        self.mock_generate.assert_called_once()
        # Check that temperature and max_tokens were passed
        call_args = self.mock_generate.call_args[1]
        assert call_args.get("temperature") == 0.5
        assert call_args.get("max_tokens") == 100

    def test_interface_compliance(self) -> None:
        """Test that ModelService implements IModelService."""
        assert isinstance(self.service, IModelService)

        # Check required methods
        assert hasattr(self.service, "get_model")
        assert hasattr(self.service, "generate")
        assert hasattr(self.service, "generate_text")
