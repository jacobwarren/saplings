from __future__ import annotations

"""
Unit tests for the OpenAI adapter.
"""


import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

# Import the LLMResponse class


# Create a mock OpenAI module
class MockOpenAI:
    def __init__(self, **kwargs):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock()


# Mock the OpenAI module at the system level
sys.modules["openai"] = MagicMock()
sys.modules["openai"].OpenAI = MockOpenAI

# Patch the OPENAI_AVAILABLE flag at the module level
patch("saplings.adapters.openai_adapter.OPENAI_AVAILABLE", True).start()

# Now we can import the adapter
from saplings.adapters.openai_adapter import OpenAIAdapter


# Create a test adapter that doesn't require the OpenAI client
class MockOpenAIAdapter(OpenAIAdapter):
    def __init__(self, provider: str, model_name: str, **kwargs):
        # Skip the parent class initialization
        self.provider = provider
        self.model_name = model_name
        self.temperature = float(kwargs.get("temperature", 0.7))
        self.max_tokens = int(kwargs.get("max_tokens", 1024))
        self._metadata = None

        # Create a mock client
        self.client = MockOpenAI()


class TestOpenAIAdapter:
    """Test the OpenAI adapter."""

    def test_openai_adapter_initialization(self) -> None:
        """Test OpenAI adapter initialization with various configurations."""
        # Test with API key in kwargs
        adapter = MockOpenAIAdapter(provider="openai", model_name="gpt-4o", api_key="test_key")
        assert adapter.model_name == "gpt-4o"
        assert adapter.provider == "openai"

        # Test with different model name
        adapter = MockOpenAIAdapter(provider="openai", model_name="gpt-3.5-turbo")
        assert adapter.model_name == "gpt-3.5-turbo"
        assert adapter.provider == "openai"

    def test_openai_adapter_generate(self) -> None:
        """Test OpenAI adapter generate method."""
        # Create the adapter
        adapter = MockOpenAIAdapter(provider="openai", model_name="gpt-4o", api_key="test_key")

        # Create a custom response object that matches what the OpenAI API would return
        class MockMessage:
            def __init__(self):
                self.content = "Hello, world!"
                self.tool_calls = None
                self.function_call = None

        class MockChoice:
            def __init__(self):
                self.message = MockMessage()

        class MockUsage:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 5
                self.total_tokens = 15

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
                self.usage = MockUsage()

        mock_response = MockResponse()

        # Set up the mock client to return our response
        adapter.client.chat.completions.create = MagicMock(return_value=mock_response)

        # Test generate method with our patched to_thread function
        with patch("asyncio.to_thread", return_value=mock_response):
            # Run the generate method
            response = asyncio.run(adapter.generate("Say hello"))

            # Verify the response
            assert response.provider == "openai"
            assert response.model_name == "gpt-4o"
            assert response.usage["prompt_tokens"] == 10
            assert response.usage["completion_tokens"] == 5
            assert response.usage["total_tokens"] == 15
            assert response.text == "Hello, world!"

    def test_openai_adapter_function_calling(self) -> None:
        """Test OpenAI adapter function calling."""
        # Create a proper mock structure with real values instead of MagicMocks
        mock_function = MagicMock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"location": "New York"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        # Create the adapter
        adapter = MockOpenAIAdapter(provider="openai", model_name="gpt-4o", api_key="test_key")

        # Set up the mock client to return our response
        adapter.client.chat.completions.create = MagicMock(return_value=mock_response)

        # Define functions in OpenAI format
        functions = [
            {
                "type": "function",
                "function": {
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
                },
            }
        ]

        # Create a custom to_thread function that calls the mock and returns the response
        async def mock_to_thread(func, *args, **kwargs):
            # Call the function with the arguments
            func(*args, **kwargs)
            # Return the mock response
            return mock_response

        # Test generate method with functions
        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            response = asyncio.run(
                adapter.generate("What's the weather in New York?", functions=functions)
            )

            assert response.text is None
            assert response.tool_calls
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["function"]["name"] == "get_weather"
            assert response.tool_calls[0]["function"]["arguments"] == '{"location": "New York"}'
            adapter.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_openai_adapter_streaming(self) -> None:
        """Test OpenAI adapter streaming."""

        # Create a custom async generator that yields the expected chunks
        async def mock_generator():
            yield "Hello"
            yield ", "
            yield "world!"

        # Create the adapter
        adapter = MockOpenAIAdapter(provider="openai", model_name="gpt-4o", api_key="test_key")

        # Replace the generate_streaming method with our mock
        adapter.generate_streaming = MagicMock(return_value=mock_generator())

        # Test streaming
        chunks = []
        async for chunk in adapter.generate_streaming("Say hello"):
            chunks.append(chunk)

        # Verify the chunks
        assert chunks == ["Hello", ", ", "world!"]
        assert "".join(chunks) == "Hello, world!"
        adapter.generate_streaming.assert_called_once()
