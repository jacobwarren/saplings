from __future__ import annotations

"""
Unit tests for the OpenAI adapter.
"""


import asyncio
from unittest.mock import MagicMock, patch

import pytest

from saplings.adapters.openai_adapter import OpenAIAdapter


class TestOpenAIAdapter:
    """Test the OpenAI adapter."""

    def test_openai_adapter_initialization(self) -> None:
        """Test OpenAI adapter initialization with various configurations."""
        # Test with API key in kwargs
        with patch("saplings.adapters.openai_adapter.OpenAI") as mock_openai:
            adapter = OpenAIAdapter(provider="openai", model_name="gpt-4o", api_key="test_key")
            assert adapter.model_name == "gpt-4o"
            assert adapter.provider == "openai"
            mock_openai.assert_called_once()

        # Test with API key from config service
        with patch("saplings.adapters.openai_adapter.OpenAI") as mock_openai:
            with patch("saplings.adapters.openai_adapter.config_service") as mock_config:
                mock_config.get_value.return_value = "config_key"
                adapter = OpenAIAdapter(provider="openai", model_name="gpt-4o")
                assert adapter.model_name == "gpt-4o"
                mock_openai.assert_called_once()
                mock_config.get_value.assert_called_with("OPENAI_API_KEY")

    def test_openai_adapter_generate(self) -> None:
        """Test OpenAI adapter generate method."""
        # Create mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello, world!"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_client.chat.completions.create.return_value = mock_response

        # Create adapter with mock client
        adapter = OpenAIAdapter(provider="openai", model_name="gpt-4o", api_key="test_key")
        adapter.client = mock_client

        # Test generate method
        response = asyncio.run(adapter.generate("Say hello"))
        assert response.text == "Hello, world!"
        assert response.provider == "openai"
        assert response.model_name == "gpt-4o"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5
        assert response.usage["total_tokens"] == 15
        mock_client.chat.completions.create.assert_called_once()

    def test_openai_adapter_function_calling(self) -> None:
        """Test OpenAI adapter function calling."""
        # Create mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=None,
                    tool_calls=[
                        MagicMock(
                            id="call_123",
                            type="function",
                            function=MagicMock(
                                name="get_weather", arguments='{"location": "New York"}'
                            ),
                        )
                    ],
                )
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_client.chat.completions.create.return_value = mock_response

        # Create adapter with mock client
        adapter = OpenAIAdapter(provider="openai", model_name="gpt-4o", api_key="test_key")
        adapter.client = mock_client

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

        # Test generate method with functions
        response = asyncio.run(
            adapter.generate("What's the weather in New York?", functions=functions)
        )

        assert response.text is None
        assert response.tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.tool_calls[0]["function"]["arguments"] == '{"location": "New York"}'
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_openai_adapter_streaming(self) -> None:
        """Test OpenAI adapter streaming."""
        # Create mock client and streaming response
        mock_client = MagicMock()
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock(delta=MagicMock(content=", "))]
        mock_chunk3 = MagicMock()
        mock_chunk3.choices = [MagicMock(delta=MagicMock(content="world!"))]

        # Set up the mock to return an async iterator
        mock_client.chat.completions.create.return_value.__aiter__.return_value = [
            mock_chunk1,
            mock_chunk2,
            mock_chunk3,
        ]

        # Create adapter with mock client
        adapter = OpenAIAdapter(provider="openai", model_name="gpt-4o", api_key="test_key")
        adapter.client = mock_client

        # Test streaming
        chunks = []
        async for chunk in adapter.generate_stream("Say hello"):
            if chunk.text:
                chunks.append(chunk.text)

        assert chunks == ["Hello", ", ", "world!"]
        assert "".join(chunks) == "Hello, world!"
        mock_client.chat.completions.create.assert_called_once()
