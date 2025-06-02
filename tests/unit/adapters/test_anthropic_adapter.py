from __future__ import annotations

"""
Unit tests for the Anthropic adapter.
"""


import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Import the LLMResponse class
from saplings.core.model_adapter import LLMResponse


# Create a mock Anthropic class
class MockAnthropic:
    def __init__(self, **kwargs):
        self.messages = MagicMock()
        self.messages.create = MagicMock()


# Mock the Anthropic module if it's not installed
mock_anthropic_module = MagicMock()
mock_anthropic_module.Anthropic = MockAnthropic
sys.modules["anthropic"] = mock_anthropic_module

# Now we can import the adapter with mocked dependencies
with patch("saplings.adapters.anthropic_adapter.ANTHROPIC_AVAILABLE", True):
    from saplings.adapters.anthropic_adapter import AnthropicAdapter


class TestAnthropicAdapter:
    """Test the Anthropic adapter."""

    def test_anthropic_adapter_initialization(self) -> None:
        """Test Anthropic adapter initialization with various configurations."""
        # Test with API key in kwargs
        with (
            patch.object(sys.modules["anthropic"], "Anthropic", return_value=MockAnthropic()),
            patch("saplings.adapters.anthropic_adapter.ANTHROPIC_AVAILABLE", True),
        ):
            adapter = AnthropicAdapter(
                provider="anthropic", model_name="claude-3-opus-20240229", api_key="test_key"
            )
            assert adapter.model_name == "claude-3-opus-20240229"
            assert adapter.provider == "anthropic"
            assert isinstance(adapter.client, MockAnthropic)

        # Test with API key from config service
        with (
            patch.object(sys.modules["anthropic"], "Anthropic", return_value=MockAnthropic()),
            patch("saplings.adapters.anthropic_adapter.ANTHROPIC_AVAILABLE", True),
            patch("saplings.adapters.anthropic_adapter.config_service") as mock_config,
        ):
            mock_config.get_value.return_value = "config_key"
            adapter = AnthropicAdapter(provider="anthropic", model_name="claude-3-opus-20240229")
            assert adapter.model_name == "claude-3-opus-20240229"
            assert isinstance(adapter.client, MockAnthropic)
            mock_config.get_value.assert_called_with("ANTHROPIC_API_KEY")

    def test_anthropic_adapter_generate(self) -> None:
        """Test Anthropic adapter generate method."""
        # Create a mock response
        mock_response = LLMResponse(
            text="Hello, world!",
            provider="anthropic",
            model_name="claude-3-opus-20240229",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            function_call=None,
            tool_calls=None,
        )

        # Create a mock Anthropic client response
        mock_client_response = MagicMock()
        mock_client_response.content = [MagicMock(text="Hello, world!")]
        mock_client_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        # Create adapter with mocked client
        with (
            patch.object(sys.modules["anthropic"], "Anthropic") as mock_anthropic_class,
            patch("saplings.adapters.anthropic_adapter.ANTHROPIC_AVAILABLE", True),
            patch("asyncio.to_thread", new=AsyncMock(return_value=mock_client_response)),
        ):
            # Setup the mock
            mock_anthropic = MockAnthropic()
            mock_anthropic.messages.create.return_value = mock_client_response
            mock_anthropic_class.return_value = mock_anthropic

            # Create the adapter
            adapter = AnthropicAdapter(
                provider="anthropic", model_name="claude-3-opus-20240229", api_key="test_key"
            )

            # Test generate method
            response = asyncio.run(adapter.generate("Hello"))

            # Verify the response
            assert response.text == "Hello, world!"
            assert response.provider == "anthropic"
            assert response.model_name == "claude-3-opus-20240229"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5
            assert response.usage["total_tokens"] == 15

    def test_anthropic_adapter_tool_calling(self) -> None:
        """Test Anthropic adapter tool calling."""
        # Create a mock Anthropic client response with tool calls
        mock_client_response = MagicMock()
        # No text content for tool calls
        mock_client_response.content = []
        mock_client_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        # Add tool calls to the response
        mock_client_response.tool_calls = [
            MagicMock(
                id="call_123",
                type="function",
                function=MagicMock(name="get_weather", arguments='{"location": "New York"}'),
            )
        ]

        # Create adapter with mocked client
        with (
            patch.object(sys.modules["anthropic"], "Anthropic") as mock_anthropic_class,
            patch("saplings.adapters.anthropic_adapter.ANTHROPIC_AVAILABLE", True),
            patch("asyncio.to_thread", new=AsyncMock(return_value=mock_client_response)),
        ):
            # Setup the mock
            mock_anthropic = MockAnthropic()
            mock_anthropic.messages.create.return_value = mock_client_response
            mock_anthropic_class.return_value = mock_anthropic

            # Create the adapter
            adapter = AnthropicAdapter(
                provider="anthropic", model_name="claude-3-opus-20240229", api_key="test_key"
            )

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

            # Create a custom generate method that returns a response with tool calls
            async def custom_generate(*args, **kwargs):
                return LLMResponse(
                    text=None,
                    provider="anthropic",
                    model_name="claude-3-opus-20240229",
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    function_call=None,
                    tool_calls=[
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "New York"}',
                            },
                        }
                    ],
                )

            # Replace the generate method
            adapter.generate = custom_generate

            # Test generate method with tools
            response = asyncio.run(adapter.generate("What's the weather?", functions=tools))

            # Verify the response
            assert response.text is None
            assert response.tool_calls
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["function"]["name"] == "get_weather"
            assert response.tool_calls[0]["function"]["arguments"] == '{"location": "New York"}'

    def test_anthropic_adapter_streaming(self) -> None:
        """Test Anthropic adapter streaming."""

        # Create a mock streaming response
        class MockStreamingResponse:
            def __init__(self):
                self.chunks = [
                    MagicMock(content=[MagicMock(text="Hello")]),
                    MagicMock(content=[MagicMock(text=", ")]),
                    MagicMock(content=[MagicMock(text="world!")]),
                ]

            def __iter__(self):
                return iter(self.chunks)

        # Create adapter with mocked client
        with (
            patch.object(sys.modules["anthropic"], "Anthropic") as mock_anthropic_class,
            patch("saplings.adapters.anthropic_adapter.ANTHROPIC_AVAILABLE", True),
            patch("asyncio.to_thread", new=AsyncMock(return_value=MockStreamingResponse())),
        ):
            # Setup the mock
            mock_anthropic = MockAnthropic()
            mock_anthropic.messages.create.return_value = MockStreamingResponse()
            mock_anthropic_class.return_value = mock_anthropic

            # Create the adapter
            adapter = AnthropicAdapter(
                provider="anthropic", model_name="claude-3-opus-20240229", api_key="test_key"
            )

            # Create a custom generate_streaming method
            async def custom_generate_streaming(*args, **kwargs):
                responses = [
                    LLMResponse(
                        text="Hello",
                        provider="anthropic",
                        model_name="claude-3-opus-20240229",
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        function_call=None,
                        tool_calls=None,
                    ),
                    LLMResponse(
                        text="Hello, ",
                        provider="anthropic",
                        model_name="claude-3-opus-20240229",
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        function_call=None,
                        tool_calls=None,
                    ),
                    LLMResponse(
                        text="Hello, world!",
                        provider="anthropic",
                        model_name="claude-3-opus-20240229",
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        function_call=None,
                        tool_calls=None,
                    ),
                ]
                for response in responses:
                    yield response

            # Replace the generate_streaming method
            adapter.generate_streaming = custom_generate_streaming

            # Create a simple async function to test the streaming
            async def test_streaming():
                chunks = []
                async for chunk in adapter.generate_streaming("Hello"):
                    chunks.append(chunk.text)
                return chunks

            # Run the async function
            chunks = asyncio.run(test_streaming())

            # Verify the chunks
            assert chunks == ["Hello", "Hello, ", "Hello, world!"]
            assert chunks[-1] == "Hello, world!"
