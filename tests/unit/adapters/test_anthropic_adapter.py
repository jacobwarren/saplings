from __future__ import annotations

"""
Unit tests for the Anthropic adapter.
"""


import asyncio
from unittest.mock import MagicMock, patch

import pytest

from saplings.adapters.anthropic_adapter import AnthropicAdapter

# Skip all tests if Anthropic is not installed
pytestmark = pytest.mark.skipif(
    not hasattr(AnthropicAdapter, "ANTHROPIC_AVAILABLE")
    or not AnthropicAdapter.ANTHROPIC_AVAILABLE,
    reason="Anthropic not installed",
)


class TestAnthropicAdapter:
    """Test the Anthropic adapter."""

    def test_anthropic_adapter_initialization(self) -> None:
        """Test Anthropic adapter initialization with various configurations."""
        # Test with API key in kwargs
        with patch("saplings.adapters.anthropic_adapter.Anthropic") as mock_anthropic:
            adapter = AnthropicAdapter(
                provider="anthropic", model_name="claude-3-opus-20240229", api_key="test_key"
            )
            assert adapter.model_name == "claude-3-opus-20240229"
            assert adapter.provider == "anthropic"
            mock_anthropic.assert_called_once()

        # Test with API key from config service
        with patch("saplings.adapters.anthropic_adapter.Anthropic") as mock_anthropic:
            with patch("saplings.adapters.anthropic_adapter.config_service") as mock_config:
                mock_config.get_value.return_value = "config_key"
                adapter = AnthropicAdapter(
                    provider="anthropic", model_name="claude-3-opus-20240229"
                )
                assert adapter.model_name == "claude-3-opus-20240229"
                mock_anthropic.assert_called_once()
                mock_config.get_value.assert_called_with("ANTHROPIC_API_KEY")

    def test_anthropic_adapter_generate(self) -> None:
        """Test Anthropic adapter generate method."""
        # Create mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello, world!")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        # Create adapter with mock client
        adapter = AnthropicAdapter(
            provider="anthropic", model_name="claude-3-opus-20240229", api_key="test_key"
        )
        adapter.client = mock_client

        # Test generate method
        response = asyncio.run(adapter.generate("Say hello"))
        assert response.text == "Hello, world!"
        assert response.provider == "anthropic"
        assert response.model_name == "claude-3-opus-20240229"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5
        assert response.usage["total_tokens"] == 15
        mock_client.messages.create.assert_called_once()

    def test_anthropic_adapter_tool_calling(self) -> None:
        """Test Anthropic adapter tool calling."""
        # Create mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_response.tool_calls = [
            MagicMock(
                id="call_123",
                type="function",
                function=MagicMock(name="get_weather", arguments='{"location": "New York"}'),
            )
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        # Create adapter with mock client
        adapter = AnthropicAdapter(
            provider="anthropic", model_name="claude-3-opus-20240229", api_key="test_key"
        )
        adapter.client = mock_client

        # Define tools
        tools = [
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

        # Test generate method with tools
        response = asyncio.run(adapter.generate("What's the weather in New York?", tools=tools))

        assert response.text is None
        assert response.tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.tool_calls[0]["function"]["arguments"] == '{"location": "New York"}'
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_anthropic_adapter_streaming(self) -> None:
        """Test Anthropic adapter streaming."""
        # Create mock client and streaming response
        mock_client = MagicMock()
        mock_chunk1 = MagicMock()
        mock_chunk1.delta = MagicMock(text="Hello")
        mock_chunk2 = MagicMock()
        mock_chunk2.delta = MagicMock(text=", ")
        mock_chunk3 = MagicMock()
        mock_chunk3.delta = MagicMock(text="world!")

        # Set up the mock to return an async iterator
        mock_client.messages.create.return_value.__aiter__.return_value = [
            mock_chunk1,
            mock_chunk2,
            mock_chunk3,
        ]

        # Create adapter with mock client
        adapter = AnthropicAdapter(
            provider="anthropic", model_name="claude-3-opus-20240229", api_key="test_key"
        )
        adapter.client = mock_client

        # Test streaming
        chunks = []
        async for chunk in adapter.generate_stream("Say hello"):
            if chunk.text:
                chunks.append(chunk.text)

        assert chunks == ["Hello", ", ", "world!"]
        assert "".join(chunks) == "Hello, world!"
        mock_client.messages.create.assert_called_once()
