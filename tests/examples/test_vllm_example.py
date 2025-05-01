"""
Tests for the vLLM example.

This module provides tests for the vLLM example code.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from examples.vllm_example import run_provider_comparison, run_vllm_example


class TestVLLMExample:
    """Test class for the vLLM example."""

    @pytest.mark.asyncio
    async def test_run_vllm_example(self, monkeypatch):
        """Test running the vLLM example."""
        # Mock the LLM class
        mock_model = MagicMock()

        # Create a mock response that can be awaited
        mock_response = MagicMock(
            text="This is a test response.",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        # Make generate return a coroutine that returns the mock response
        async def mock_generate(*args, **kwargs):
            return mock_response

        mock_model.generate = mock_generate

        # Mock the streaming generator
        async def mock_streaming_generator(*args, **kwargs):
            for chunk in ["This ", "is ", "a ", "test ", "response."]:
                yield chunk

        mock_model.generate_streaming = mock_streaming_generator
        mock_model.cleanup = MagicMock()

        # Mock the LLM.from_uri method
        with patch("examples.vllm_example.LLM.from_uri", return_value=mock_model):
            # Run the example
            await run_vllm_example("test-model")

            # Check that cleanup was called
            mock_model.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_provider_comparison(self, monkeypatch):
        """Test running the provider comparison."""
        # Mock the LLM class
        mock_vllm_model = MagicMock()
        mock_vllm_response = MagicMock(
            text="vLLM response",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        async def mock_vllm_generate(*args, **kwargs):
            return mock_vllm_response

        mock_vllm_model.generate = mock_vllm_generate
        mock_vllm_model.cleanup = MagicMock()

        mock_openai_model = MagicMock()
        mock_openai_response = MagicMock(
            text="OpenAI response",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        async def mock_openai_generate(*args, **kwargs):
            return mock_openai_response

        mock_openai_model.generate = mock_openai_generate

        mock_anthropic_model = MagicMock()
        mock_anthropic_response = MagicMock(
            text="Anthropic response",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        async def mock_anthropic_generate(*args, **kwargs):
            return mock_anthropic_response

        mock_anthropic_model.generate = mock_anthropic_generate

        mock_huggingface_model = MagicMock()
        mock_huggingface_response = MagicMock(
            text="HuggingFace response",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        async def mock_huggingface_generate(*args, **kwargs):
            return mock_huggingface_response

        mock_huggingface_model.generate = mock_huggingface_generate
        mock_huggingface_model.cleanup = MagicMock()

        # Add hasattr method to all models to properly handle cleanup checks
        def mock_hasattr(self, attr):
            return attr == "cleanup"

        type(mock_vllm_model).__hasattr__ = mock_hasattr
        type(mock_openai_model).__hasattr__ = lambda self, attr: False
        type(mock_anthropic_model).__hasattr__ = lambda self, attr: False
        type(mock_huggingface_model).__hasattr__ = mock_hasattr

        # Mock the LLM.from_uri method
        def mock_from_uri(uri):
            if uri.startswith("vllm://"):
                return mock_vllm_model
            elif uri.startswith("openai://"):
                return mock_openai_model
            elif uri.startswith("anthropic://"):
                return mock_anthropic_model
            elif uri.startswith("huggingface://"):
                return mock_huggingface_model
            else:
                raise ValueError(f"Unsupported URI: {uri}")

        with patch("examples.vllm_example.LLM.from_uri", side_effect=mock_from_uri):
            # Run the example
            await run_provider_comparison()

            # We don't need to check if the functions were called
            # The test passes if it completes without errors and cleanup is called correctly

            # Check that cleanup was called for models that support it
            mock_vllm_model.cleanup.assert_called_once()
            mock_huggingface_model.cleanup.assert_called_once()
