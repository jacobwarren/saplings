from __future__ import annotations

from unittest.mock import MagicMock

from saplings.core.interfaces import IExecutionService, IGASAService
from saplings.core.model_adapter import LLMResponse
from saplings.executor.config import ExecutorConfig, RefinementStrategy, ValidationStrategy
from saplings.services.execution_service import ExecutionService

"""
Unit tests for the execution service.
"""


class TestExecutionService:
    THRESHOLD_1 = 0.7

    """Test the execution service."""

    def setup_method(self) -> None:
        """Set up test environment."""
        # Create mock model
        self.mock_model = MagicMock()

        # Mock the model response
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.text = "This is a test response."
        mock_response.provider = "test"
        mock_response.model_name = "test-model"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        self.mock_model.generate.return_value = mock_response

        # Mock the streaming response
        from unittest.mock import AsyncMock

        self.mock_model.generate_stream = AsyncMock()
        mock_chunk1 = MagicMock(spec=LLMResponse)
        mock_chunk1.text = "This "
        mock_chunk2 = MagicMock(spec=LLMResponse)
        mock_chunk2.text = "is "
        mock_chunk3 = MagicMock(spec=LLMResponse)
        mock_chunk3.text = "a test response."
        mock_aiter = MagicMock()
        self.mock_model.generate_stream.return_value = mock_aiter
        mock_aiter.__aiter__ = MagicMock()
        mock_aiter.__aiter__.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]

        # Create mock GASA service
        self.mock_gasa_service = MagicMock(spec=IGASAService)
        self.mock_gasa_service.apply_gasa.return_value = {
            "prompt": "This is a test prompt with context.",
            "attention_mask": None,
        }

        # Create execution service
        self.config = ExecutorConfig(
            execution_model_provider="test",
            execution_model_name="test-model",
            max_tokens=1024,
            temperature=0.7,
            enable_gasa=True,
            enable_speculative_execution=False,
            draft_temperature=0.0,
            final_temperature=0.0,
            max_draft_tokens=0,
            max_final_tokens=0,
            enable_streaming=False,
            stream_chunk_size=0,
            gasa_config=None,
            validation_type=ValidationStrategy.NONE,
            validation_threshold=0.0,
            refinement_strategy=RefinementStrategy.NONE,
            max_refinement_attempts=0,
            cache_results=False,
            cache_dir=None,
            log_level="INFO",
        )
        self.service = ExecutionService(
            model=self.mock_model,
            gasa_service=self.mock_gasa_service,
            config=self.config,
        )

    def test_initialization(self) -> None:
        """Test execution service initialization."""
        assert self.service.gasa_service is self.mock_gasa_service
        assert self.service.config is self.config

        # Make sure config is not None before accessing its attributes
        if self.service.config is not None:
            assert self.service.config.execution_model_provider == "test"
            assert self.service.config.execution_model_name == "test-model"
            assert self.service.config.max_tokens == 1024
            assert self.service.config.temperature == self.THRESHOLD_1
            assert self.service.config.enable_gasa is True

    async def test_execute(self) -> None:
        """Test execute method."""
        # Execute a prompt
        response = await self.service.execute(
            prompt="This is a test prompt.", documents=["This is context 1.", "This is context 2."]
        )

        # Verify response
        assert response == self.mock_model.generate.return_value

        # Verify model was called
        self.mock_model.generate.assert_called_once()

        # Verify GASA service was called
        self.mock_gasa_service.apply_gasa.assert_called_once()

    async def test_execute_without_gasa(self) -> None:
        """Test execute method without GASA."""
        # Disable GASA
        if self.service.config is not None:
            self.service.config.enable_gasa = False
        else:
            # If config is None, create a new config with enable_gasa=False
            # Create a config with default values but with enable_gasa=False
            from saplings.executor.config import (
                ExecutorConfig,
                RefinementStrategy,
                ValidationStrategy,
            )

            self.service.config = ExecutorConfig(
                execution_model_provider="test",
                execution_model_name="test-model",
                max_tokens=1024,
                temperature=0.7,
                enable_gasa=False,
                enable_speculative_execution=False,
                draft_temperature=0.0,
                final_temperature=0.0,
                max_draft_tokens=0,
                max_final_tokens=0,
                enable_streaming=False,
                stream_chunk_size=0,
                gasa_config=None,
                validation_type=ValidationStrategy.NONE,
                validation_threshold=0.0,
                refinement_strategy=RefinementStrategy.NONE,
                max_refinement_attempts=0,
                cache_results=False,
                cache_dir=None,
                log_level="INFO",
            )

        # Execute a prompt
        response = await self.service.execute(
            prompt="This is a test prompt.", documents=["This is context 1.", "This is context 2."]
        )

        # Verify response
        assert response == self.mock_model.generate.return_value

        # Verify model was called
        self.mock_model.generate.assert_called_once()

        # Verify GASA service was not called
        self.mock_gasa_service.apply_gasa.assert_not_called()

    async def test_execute_with_parameters(self) -> None:
        """Test execute method with parameters."""
        # Execute a prompt with parameters
        response = await self.service.execute(
            prompt="This is a test prompt.",
            documents=["This is context 1.", "This is context 2."],
            temperature=0.5,
            max_tokens=500,
        )

        # Verify response
        assert response == self.mock_model.generate.return_value

        # Verify model was called with parameters
        call_kwargs = self.mock_model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500

    async def test_execute_with_functions(self) -> None:
        """Test execute method with functions."""
        # Define functions
        functions = [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for",
                        }
                    },
                    "required": ["location"],
                },
            }
        ]

        # Execute a prompt with functions
        response = await self.service.execute(
            prompt="What's the weather in New York?", documents=[], functions=functions
        )

        # Verify response
        assert response == self.mock_model.generate.return_value

        # Verify model was called with functions
        call_kwargs = self.mock_model.generate.call_args[1]
        assert "functions" in call_kwargs
        assert call_kwargs["functions"] == functions

    async def test_execute_with_system_prompt(self) -> None:
        """Test execute method with system prompt."""
        # Execute a prompt with system prompt
        response = await self.service.execute(
            prompt="This is a test prompt.",
            documents=["This is context 1.", "This is context 2."],
            system_prompt="You are a helpful assistant.",
        )

        # Verify response
        assert response == self.mock_model.generate.return_value

        # Verify model was called with system prompt
        call_kwargs = self.mock_model.generate.call_args[1]
        assert "system_prompt" in call_kwargs
        assert call_kwargs["system_prompt"] == "You are a helpful assistant."

    async def test_execute_stream(self) -> None:
        """Test execute_stream method."""
        # The chunks are already set up in setup_method

        # Execute a streaming prompt directly
        chunks = await self.service.execute_stream(
            prompt="This is a test prompt.",
            documents=["This is context 1.", "This is context 2."],
        )

        # Extract the text from each chunk
        chunk_texts = [chunk.text for chunk in chunks]

        # Verify chunks
        assert chunk_texts == ["This ", "is ", "a test response."]
        assert "".join(chunk_texts) == "This is a test response."

        # Verify model was called
        self.mock_model.generate_stream.assert_called_once()

        # Verify GASA service was called
        self.mock_gasa_service.apply_gasa.assert_called_once()

    def test_interface_compliance(self) -> None:
        """Test that ExecutionService implements IExecutionService."""
        assert isinstance(self.service, IExecutionService)

        # Check required methods
        assert hasattr(self.service, "execute")
        assert hasattr(self.service, "execute_stream")
