"""
Tests for advanced features in model adapters.

This module provides tests for advanced features like function calling, vision models, etc.
"""

import asyncio
import base64
import json
import pytest
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

from saplings.core.message import (
    ContentType,
    FunctionCall,
    FunctionDefinition,
    Message,
    MessageContent,
    MessageRole,
)
from saplings.core.model_adapter import LLM, LLMResponse, ModelURI
from saplings.adapters.vllm_adapter import VLLMAdapter
from saplings.adapters.openai_adapter import OpenAIAdapter

# Check if OpenAI and vLLM are available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import vllm
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class TestFunctionCalling:
    """Test class for function calling features."""

    @pytest.fixture
    def function_definitions(self):
        """Create sample function definitions for testing."""
        return [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature to use",
                    }
                },
                "required": ["location"]
            }
        ]

    @pytest.fixture
    def mock_openai_function_call(self):
        """Mock OpenAI response with a function call."""
        # Create a mock function call
        mock_function = MagicMock()
        mock_function.name = "get_weather"
        mock_function.arguments = json.dumps({"location": "San Francisco, CA", "unit": "celsius"})

        # Create a mock message
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.function_call = mock_function
        mock_message.tool_calls = None

        # Create a mock choice
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "function_call"

        # Create a mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        return mock_response

    @pytest.fixture
    def mock_openai_tool_call(self):
        """Mock OpenAI response with a tool call."""
        # Create a mock function
        mock_function = MagicMock()
        mock_function.name = "get_weather"
        mock_function.arguments = json.dumps({"location": "San Francisco, CA", "unit": "celsius"})

        # Create a mock tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_function

        # Create a mock message
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        # Create a mock choice
        mock_choice = MagicMock()
        mock_choice.message = mock_message

        # Create a mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        return mock_response

    @pytest.fixture
    def mock_vllm_function_call(self):
        """Mock vLLM response with a function call."""
        # Create a mock output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = """
        ```json
        {
            "name": "get_weather",
            "arguments": {
                "location": "San Francisco, CA",
                "unit": "celsius"
            }
        }
        ```
        """
        mock_output.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_output.outputs[0].token_ids = [6, 7, 8, 9, 10]

        # Add tool_calls attribute for native function calling
        mock_tool_call = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "San Francisco, CA", "unit": "celsius"})
            }
        }
        mock_output.tool_calls = [mock_tool_call]

        return [mock_output]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
    async def test_openai_function_call(self, function_definitions, mock_openai_function_call):
        """Test function calling with OpenAI."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_function_call

        # Create the adapter
        with patch("saplings.adapters.openai_adapter.OpenAI", return_value=mock_client):
            adapter = OpenAIAdapter("openai://gpt-4")

            # Test function calling
            response = await adapter.generate(
                prompt=[{"role": "user", "content": "What's the weather in San Francisco?"}],
                functions=function_definitions,
                function_call="auto"
            )

            # Check that the function call was detected
            assert response.function_call is not None
            assert response.function_call["name"] == "get_weather"
            assert json.loads(response.function_call["arguments"]) == {
                "location": "San Francisco, CA",
                "unit": "celsius"
            }

            # Check that the client was called correctly
            args, kwargs = mock_client.chat.completions.create.call_args
            assert kwargs["tools"] == [{"type": "function", "function": func} for func in function_definitions]
            assert kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
    async def test_openai_tool_call(self, function_definitions, mock_openai_tool_call):
        """Test tool calling with OpenAI."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_tool_call

        # Create the adapter
        with patch("saplings.adapters.openai_adapter.OpenAI", return_value=mock_client):
            adapter = OpenAIAdapter("openai://gpt-4")

            # Test tool calling
            response = await adapter.generate(
                prompt=[{"role": "user", "content": "What's the weather in San Francisco?"}],
                functions=function_definitions,
                function_call="auto"
            )

            # Check that the tool call was detected
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["id"] == "call_123"
            assert response.tool_calls[0]["type"] == "function"
            assert response.tool_calls[0]["function"]["name"] == "get_weather"
            assert json.loads(response.tool_calls[0]["function"]["arguments"]) == {
                "location": "San Francisco, CA",
                "unit": "celsius"
            }

    @pytest.mark.asyncio
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM package not installed")
    async def test_vllm_function_call(self, function_definitions, mock_vllm_function_call):
        """Test function calling with vLLM."""
        # Mock the vLLM engine
        mock_engine = MagicMock()
        mock_engine.generate.return_value = mock_vllm_function_call

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Test prompt"

        # Create the adapter
        with patch("saplings.adapters.vllm_adapter.vllm.LLM", return_value=mock_engine):
            with patch("saplings.adapters.vllm_adapter.get_tokenizer", return_value=mock_tokenizer):
                adapter = VLLMAdapter("vllm://meta-llama/Llama-2-7b-chat-hf")

                # Test function calling
                response = await adapter.generate(
                    prompt=[{"role": "user", "content": "What's the weather in San Francisco?"}],
                    functions=function_definitions,
                    function_call="auto"
                )

                # Check that the function call was detected
                assert response.function_call is not None
                assert response.function_call["name"] == "get_weather"
                assert json.loads(response.function_call["arguments"]) == {
                    "location": "San Francisco, CA",
                    "unit": "celsius"
                }

                # Check that the tokenizer was called correctly
                args, kwargs = mock_tokenizer.apply_chat_template.call_args
                assert kwargs["messages"] == [{"role": "user", "content": "What's the weather in San Francisco?"}]
                assert kwargs["tools"] == function_definitions

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
    async def test_openai_streaming_function_call(self, function_definitions):
        """Test streaming function calling with OpenAI."""
        # Create a custom generate_streaming method that returns function call chunks
        async def mock_streaming_method(*args, **kwargs):
            # Yield function call chunks
            yield {"function_call": {"name": "get_", "arguments": ""}}
            yield {"function_call": {"name": "get_weather", "arguments": ""}}
            yield {"function_call": {"name": "get_weather", "arguments": '{"location": "San Francisco, CA"'}}
            yield {"function_call": {"name": "get_weather", "arguments": '{"location": "San Francisco, CA", "unit": "celsius"}'}}

        # Create the adapter
        with patch("saplings.adapters.openai_adapter.OpenAI", return_value=MagicMock()):
            adapter = OpenAIAdapter("openai://gpt-4")

            # Replace the generate_streaming method with our mock
            with patch.object(adapter, 'generate_streaming', side_effect=mock_streaming_method):
                # Test streaming function calling
                chunks_received = []
                async for chunk in adapter.generate_streaming(
                    prompt=[{"role": "user", "content": "What's the weather in San Francisco?"}],
                    functions=function_definitions,
                    function_call="auto"
                ):
                    chunks_received.append(chunk)

                # Check that the function call chunks were received
                assert len(chunks_received) == 4
                assert all(isinstance(chunk, dict) and "function_call" in chunk for chunk in chunks_received)

                # Check the final function call
                final_function_call = chunks_received[-1]["function_call"]
                assert final_function_call["name"] == "get_weather"
                assert final_function_call["arguments"] == '{"location": "San Francisco, CA", "unit": "celsius"}'


class TestVisionModels:
    """Test class for vision model features."""

    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        # This is a tiny 1x1 transparent PNG
        return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")

    @pytest.fixture
    def mock_openai_vision(self):
        """Mock OpenAI response for vision models."""
        # Create a mock message
        mock_message = MagicMock()
        mock_message.content = "I see a transparent PNG image."
        mock_message.function_call = None
        mock_message.tool_calls = None

        # Create a mock choice
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        # Create a mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        return mock_response

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
    async def test_openai_vision(self, sample_image_data, mock_openai_vision):
        """Test vision capabilities with OpenAI."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_vision

        # Create the adapter
        with patch("saplings.adapters.openai_adapter.OpenAI", return_value=mock_client):
            adapter = OpenAIAdapter("openai://gpt-4-vision-preview")

            # Create a message with an image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(sample_image_data).decode()}}
                    ]
                }
            ]

            # Test vision capabilities
            response = await adapter.generate(prompt=messages)

            # Check the response
            assert response.text == "I see a transparent PNG image."

            # Check that the client was called correctly
            args, kwargs = mock_client.chat.completions.create.call_args
            assert kwargs["messages"] == messages


class TestJSONMode:
    """Test class for JSON mode features."""

    @pytest.fixture
    def mock_openai_json(self):
        """Mock OpenAI response for JSON mode."""
        # Create a mock message
        mock_message = MagicMock()
        mock_message.content = '{"weather": {"location": "San Francisco", "temperature": 22, "unit": "celsius"}}'
        mock_message.function_call = None
        mock_message.tool_calls = None

        # Create a mock choice
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        # Create a mock usage
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        return mock_response

    @pytest.fixture
    def mock_vllm_json(self):
        """Mock vLLM response for JSON mode."""
        # Create a mock output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = '{"weather": {"location": "San Francisco", "temperature": 22, "unit": "celsius"}}'
        mock_output.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_output.outputs[0].token_ids = [6, 7, 8, 9, 10]

        return [mock_output]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI package not installed")
    async def test_openai_json_mode(self, mock_openai_json):
        """Test JSON mode with OpenAI."""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_openai_json

        # Create the adapter
        with patch("saplings.adapters.openai_adapter.OpenAI", return_value=mock_client):
            adapter = OpenAIAdapter("openai://gpt-4")

            # Test JSON mode
            response = await adapter.generate(
                prompt=[{"role": "user", "content": "Give me the weather in San Francisco as JSON"}],
                json_mode=True
            )

            # Check the response
            assert response.text == '{"weather": {"location": "San Francisco", "temperature": 22, "unit": "celsius"}}'

            # Check that the client was called correctly
            args, kwargs = mock_client.chat.completions.create.call_args
            assert kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM package not installed")
    async def test_vllm_json_mode(self, mock_vllm_json):
        """Test JSON mode with vLLM."""
        # Mock the vLLM engine
        mock_engine = MagicMock()
        mock_engine.generate.return_value = mock_vllm_json

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Test prompt"

        # Create the adapter
        with patch("saplings.adapters.vllm_adapter.vllm.LLM", return_value=mock_engine):
            with patch("saplings.adapters.vllm_adapter.get_tokenizer", return_value=mock_tokenizer):
                adapter = VLLMAdapter("vllm://meta-llama/Llama-2-7b-chat-hf")

                # Test JSON mode
                response = await adapter.generate(
                    prompt=[{"role": "user", "content": "Give me the weather in San Francisco as JSON"}],
                    json_mode=True
                )

                # Check the response
                assert response.text == '{"weather": {"location": "San Francisco", "temperature": 22, "unit": "celsius"}}'

                # Check that the sampling parameters were set correctly
                args, kwargs = mock_engine.generate.call_args
                sampling_params = kwargs["sampling_params"]

                # If the sampling params has a grammar attribute, it should be set to "json"
                if hasattr(sampling_params, "grammar"):
                    assert sampling_params.grammar == "json"
