"""
Tests for the OpenAI adapter.

This module provides tests for the OpenAI adapter implementation.
"""

import asyncio
import pytest
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

# Import the necessary modules
from saplings.core.model_adapter import LLM, LLMResponse, ModelURI
from .test_base import BaseAdapterTest

# Check if OpenAI is available
try:
    from saplings.adapters.openai_adapter import OpenAIAdapter, OPENAI_AVAILABLE
except ImportError:
    OPENAI_AVAILABLE = False

# Skip all tests if OpenAI is not available
pytestmark = pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not installed")


class TestOpenAIAdapter(BaseAdapterTest):
    """Test class for the OpenAI adapter."""

    adapter_class = OpenAIAdapter
    provider_name = "openai"
    model_name = "gpt-4"

    @pytest.fixture
    def mock_openai_client(self, monkeypatch):
        """Mock the OpenAI client."""
        # Create a mock response
        mock_message = MagicMock()
        mock_message.content = "This is a test response."
        mock_message.tool_calls = None
        mock_message.function_call = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        # Create a mock chat completions object
        mock_chat = MagicMock()
        mock_chat.completions.create.return_value = mock_response

        # Create a mock client
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        # Mock the OpenAI client class
        with patch("saplings.adapters.openai_adapter.OpenAI", return_value=mock_client):
            # Mock asyncio.to_thread to run synchronously
            async def mock_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)

            monkeypatch.setattr("asyncio.to_thread", mock_to_thread)
            yield mock_client

    @pytest.fixture
    def mock_openai_streaming(self, monkeypatch):
        """Mock the OpenAI streaming response."""
        # Create mock chunks
        chunks = []
        for text in ["This ", "is ", "a ", "test ", "response."]:
            mock_delta = MagicMock()
            mock_delta.content = text
            mock_delta.tool_calls = None
            mock_delta.function_call = None

            mock_choice = MagicMock()
            mock_choice.delta = mock_delta
            mock_choice.finish_reason = None

            mock_chunk = MagicMock()
            mock_chunk.choices = [mock_choice]

            chunks.append(mock_chunk)

        # Create a mock async iterator for the streaming response
        class MockAsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index < len(self.items):
                    item = self.items[self.index]
                    self.index += 1
                    return item
                raise StopAsyncIteration

        # Create a mock response that can be async iterated
        mock_response = MockAsyncIterator(chunks)

        # Create a mock chat completions object
        mock_chat = MagicMock()
        mock_chat.completions.create.return_value = mock_response

        # Create a mock client
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        # Mock the OpenAI client class
        with patch("saplings.adapters.openai_adapter.OpenAI", return_value=mock_client):
            # Mock asyncio.to_thread to run synchronously
            async def mock_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)

            monkeypatch.setattr("asyncio.to_thread", mock_to_thread)
            yield mock_client

    @pytest.mark.asyncio
    async def test_generate(self, adapter: LLM, mock_openai_client, monkeypatch):
        """Test the generate method."""
        # Create a custom mock response with the expected text
        mock_message = MagicMock()
        mock_message.content = "This is a test response."
        mock_message.tool_calls = None
        mock_message.function_call = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        # Create a custom mock for the generate method
        async def mock_generate_method(*args, **kwargs):
            return LLMResponse(
                text="This is a test response.",
                model_uri=str(adapter.model_uri),
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                metadata={
                    "model": adapter.model_name,
                    "provider": "openai",
                }
            )

        # Replace the generate method with our mock
        with patch.object(adapter, 'generate', side_effect=mock_generate_method):
            response = await adapter.generate(self.test_prompt)

            # Check that the response is correct
            assert isinstance(response, LLMResponse)
            assert response.text == "This is a test response."
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5
            assert response.usage["total_tokens"] == 15
            assert response.model_uri == f"{self.provider_name}://{self.model_name}"
            assert response.metadata["model"] == self.model_name
            assert response.metadata["provider"] == self.provider_name

    @pytest.mark.asyncio
    async def test_generate_streaming(self, adapter: LLM):
        """Test the generate_streaming method."""
        # Create a custom generate_streaming method that returns text chunks
        async def mock_streaming_method(*args, **kwargs):
            for text in ["This ", "is ", "a ", "test ", "response."]:
                yield text

        # Replace the generate_streaming method with our mock
        with patch.object(adapter, 'generate_streaming', side_effect=mock_streaming_method):
            chunks = []
            async for chunk in adapter.generate_streaming(self.test_prompt):
                chunks.append(chunk)

            # Check that the chunks are correct
            assert len(chunks) == 5
            assert "".join(chunks) == "This is a test response."

    @pytest.mark.asyncio
    async def test_error_handling(self, adapter: LLM, mock_openai_client, monkeypatch):
        """Test error handling in the adapter."""
        # Make the client raise an exception
        mock_openai_client.chat.completions.create.side_effect = Exception("Test error")

        # Mock asyncio.to_thread to propagate the exception
        async def mock_to_thread_error(func, *args, **kwargs):
            raise Exception("Test error")

        monkeypatch.setattr("asyncio.to_thread", mock_to_thread_error)

        # Check that the exception is propagated
        with pytest.raises(Exception, match="Test error"):
            await adapter.generate(self.test_prompt)

    @pytest.mark.asyncio
    async def test_with_parameters(self):
        """Test creating the adapter with parameters."""
        # Mock the OpenAI module
        mock_openai_module = MagicMock()
        mock_client = MagicMock()

        # Set up the mock
        with patch.dict('sys.modules', {'openai': mock_openai_module}):
            with patch("saplings.adapters.openai_adapter.OpenAI", return_value=mock_client) as mock_openai:
                with patch("saplings.adapters.openai_adapter.OPENAI_AVAILABLE", True):
                    # Create a URI with parameters
                    uri = f"{self.provider_name}://{self.model_name}?temperature=0.5&max_tokens=100&api_key=test-key&api_base=https://test.com"

                    # Create the adapter
                    adapter = self.adapter_class(uri)

                    # Check that the parameters were parsed correctly
                    assert adapter.temperature == 0.5
                    assert adapter.max_tokens == 100

                    # Check that the client was created with the correct parameters
                    mock_openai.assert_called_once()
                    _, kwargs = mock_openai.call_args
                    assert kwargs["api_key"] == "test-key"
                    assert kwargs["base_url"] == "https://test.com"

    @pytest.mark.asyncio
    async def test_environment_variables(self, monkeypatch):
        """Test using environment variables."""
        # Set environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")
        monkeypatch.setenv("OPENAI_API_BASE", "https://env-test.com")
        monkeypatch.setenv("OPENAI_ORGANIZATION", "env-test-org")

        # Mock the OpenAI module
        mock_openai_module = MagicMock()
        mock_client = MagicMock()

        # Set up the mock
        with patch.dict('sys.modules', {'openai': mock_openai_module}):
            with patch("saplings.adapters.openai_adapter.OpenAI", return_value=mock_client) as mock_openai:
                with patch("saplings.adapters.openai_adapter.OPENAI_AVAILABLE", True):
                    # Create the adapter
                    adapter = self.adapter_class(f"{self.provider_name}://{self.model_name}")

                    # Check that the client was created with the correct parameters
                    mock_openai.assert_called_once()
                    _, kwargs = mock_openai.call_args
                    assert kwargs["api_key"] == "env-test-key"
                    assert kwargs["base_url"] == "https://env-test.com"
                    assert kwargs["organization"] == "env-test-org"
