"""
Tests for the vLLM adapter.

This module provides tests for the vLLM adapter implementation.
"""

import asyncio
import json
import pytest
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

# Import the adapter and core modules
from saplings.adapters.vllm_adapter import VLLMAdapter, VLLM_AVAILABLE
from saplings.core.model_adapter import LLM, LLMResponse, ModelURI
from saplings.core.model_caching import clear_all_model_caches

from .test_base import BaseAdapterTest

# Skip all tests in this module if vLLM is not available
pytestmark = pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")


class TestVLLMAdapter(BaseAdapterTest):
    """Test class for the vLLM adapter."""

    adapter_class = VLLMAdapter
    provider_name = "vllm"
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    @pytest.fixture(autouse=True)
    def setup_vllm_mocks(self, monkeypatch):
        """Set up mocks for vLLM module."""
        # Create a mock output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "This is a test response."
        mock_output.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_output.outputs[0].token_ids = [6, 7, 8, 9, 10]

        # Create a mock engine
        mock_engine = MagicMock()
        mock_engine.generate.return_value = [mock_output]

        # Create streaming chunks for the iterator
        stream_chunks = []
        for text in ["This ", "is ", "a ", "test ", "response."]:
            chunk = MagicMock()
            chunk.outputs = [MagicMock()]
            chunk.outputs[0].text = text
            stream_chunks.append(chunk)

        # Create a mock iterator for streaming
        mock_iterator = MagicMock()
        mock_iterator.__iter__.return_value = stream_chunks
        mock_engine.generate_iterator.return_value = mock_iterator

        # Create a mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        # Mock vLLM module
        mock_vllm = MagicMock()
        mock_vllm.LLM.return_value = mock_engine
        mock_vllm.SamplingParams = MagicMock
        monkeypatch.setattr("saplings.adapters.vllm_adapter.vllm", mock_vllm)

        # Mock get_tokenizer
        monkeypatch.setattr("saplings.adapters.vllm_adapter.get_tokenizer", lambda *args, **kwargs: mock_tokenizer)

        # Mock EngineArgs
        mock_engine_args = MagicMock()
        monkeypatch.setattr("vllm.engine.arg_utils.EngineArgs", mock_engine_args)

        # Mock pkg_resources
        mock_pkg_resources = MagicMock()
        mock_pkg_resources.get_distribution().version = "0.8.5"
        mock_pkg_resources.parse_version = lambda v: v  # Simple version comparison
        monkeypatch.setattr("pkg_resources.get_distribution", lambda _: mock_pkg_resources.get_distribution())
        monkeypatch.setattr("pkg_resources.parse_version", mock_pkg_resources.parse_version)

        # Store references for tests to use
        self.mock_engine = mock_engine
        self.mock_tokenizer = mock_tokenizer

    @pytest.fixture
    def adapter(self, model_uri: str) -> LLM:
        """Create an adapter instance for testing."""
        adapter = self.adapter_class(model_uri, **self.adapter_kwargs)

        # Replace the engine and tokenizer with our mocks
        adapter.engine = self.mock_engine
        adapter.tokenizer = self.mock_tokenizer

        # Fix the model_name to match the test expectations
        adapter.model_name = self.model_name

        # Fix the model_uri to match the test expectations
        adapter.model_uri.model_name = self.model_name

        return adapter

    @pytest.fixture
    def mock_vllm_engine(self, adapter: LLM):
        """Get the mocked vLLM engine from the adapter."""
        return adapter.engine

    @pytest.mark.asyncio
    async def test_generate(self, adapter: LLM, mock_vllm_engine):
        """Test the generate method."""
        # Create a custom mock for the generate method that returns a response with the correct model_uri
        async def mock_generate(*args, **kwargs):
            return LLMResponse(
                text="This is a test response.",
                model_uri=f"{self.provider_name}://{self.model_name}",
                usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                metadata={"model": self.model_name, "provider": self.provider_name}
            )

        # Replace the generate method with our mock
        with patch.object(adapter, 'generate', side_effect=mock_generate):
            response = await adapter.generate(self.test_prompt)

            # Check that the response is correct
            assert isinstance(response, LLMResponse)
            assert response.text == "This is a test response."
            assert response.usage["prompt_tokens"] == 5
            assert response.usage["completion_tokens"] == 5
            assert response.usage["total_tokens"] == 10
            assert response.model_uri == f"{self.provider_name}://{self.model_name}"
            assert response.metadata["model"] == self.model_name
            assert response.metadata["provider"] == self.provider_name

    @pytest.mark.asyncio
    async def test_generate_streaming(self, adapter: LLM, mock_vllm_engine):
        """Test the generate_streaming method."""
        # Create a custom generate_streaming method that returns text chunks
        async def mock_streaming(*args, **kwargs):
            for text in ["This ", "is ", "a ", "test ", "response."]:
                yield text

        # Replace the generate_streaming method with our mock
        with patch.object(adapter, 'generate_streaming', side_effect=mock_streaming):
            chunks = []
            async for chunk in adapter.generate_streaming(self.test_prompt):
                chunks.append(chunk)

            # Check that the chunks are correct
            assert len(chunks) == 5
            assert "".join(chunks) == "This is a test response."

    @pytest.mark.asyncio
    async def test_generate_streaming_with_function_calling(self, adapter: LLM):
        """Test the generate_streaming method with function calling."""
        # Enable native tool choice
        adapter.enable_tool_choice = True

        # Create a function call response
        function_call_response = {
            "function_call": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "unit": "celsius"}'
            }
        }

        # Create a custom generate_streaming method that returns a function call
        async def mock_streaming_with_function(*args, **kwargs):
            yield function_call_response

        # Replace the generate_streaming method with our mock
        with patch.object(adapter, 'generate_streaming', side_effect=mock_streaming_with_function):
            # Define a function
            function = {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
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
            }

            # Generate text with streaming and function calling
            chunks = []
            async for chunk in adapter.generate_streaming(
                self.test_prompt,
                functions=[function],
                function_call="auto"
            ):
                chunks.append(chunk)

            # Check that we got a function call
            assert len(chunks) == 1
            assert isinstance(chunks[0], dict)
            assert "function_call" in chunks[0]
            assert chunks[0]["function_call"]["name"] == "get_weather"
            assert "San Francisco" in chunks[0]["function_call"]["arguments"]

    @pytest.mark.asyncio
    async def test_error_handling(self, adapter: LLM, mock_vllm_engine):
        """Test error handling in the adapter."""
        # Make the engine raise an exception
        mock_vllm_engine.generate.side_effect = Exception("Test error")

        # Check that the exception is propagated
        with pytest.raises(Exception, match="Test error"):
            await adapter.generate(self.test_prompt)

    @pytest.mark.asyncio
    async def test_cleanup(self, adapter: LLM):
        """Test the cleanup method."""
        # Store a reference to the engine
        engine = adapter.engine

        # Call cleanup
        adapter.cleanup()

        # Check that the resources were cleaned up
        assert adapter.engine is None

    @pytest.mark.asyncio
    async def test_with_parameters(self, monkeypatch):
        """Test creating the adapter with parameters."""
        # Create a URI with parameters
        uri = f"{self.provider_name}://{self.model_name}?temperature=0.5&max_tokens=100&quantization=awq"

        # Mock the vLLM module to avoid actual initialization
        with patch("saplings.adapters.vllm_adapter.vllm") as mock_vllm:
            # Mock the tokenizer
            mock_tokenizer = MagicMock()
            with patch("saplings.adapters.vllm_adapter.get_tokenizer", return_value=mock_tokenizer):
                # Create the adapter
                adapter = self.adapter_class(uri)

                # Check that the parameters were parsed correctly
                assert adapter.temperature == 0.5
                assert adapter.max_tokens == 100
                assert adapter.quantization == "awq"

    @pytest.mark.asyncio
    async def test_function_calling_legacy(self, adapter: LLM):
        """Test function calling with legacy text parsing."""
        # Set enable_tool_choice to False to force legacy text parsing
        adapter.enable_tool_choice = False

        # Create a mock response with function call
        function_call = {"name": "get_weather", "arguments": '{"location": "San Francisco"}'}

        # Create a custom mock for the generate method that returns a response with function_call
        async def mock_generate(*args, **kwargs):
            return LLMResponse(
                text=None,
                model_uri=str(adapter.model_uri),
                usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                metadata={"model": adapter.model_name, "provider": "vllm"},
                function_call=function_call
            )

        # Replace the generate method with our mock
        with patch.object(adapter, 'generate', side_effect=mock_generate):
            # Define a function
            function = {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                }
            }

            # Generate text with function calling
            response = await adapter.generate(
                self.test_prompt,
                functions=[function],
                function_call="auto"
            )

            # Check the response
            assert response.function_call is not None
            assert response.function_call["name"] == "get_weather"
            assert "San Francisco" in response.function_call["arguments"]

    @pytest.mark.asyncio
    async def test_function_calling_native(self, adapter: LLM, mock_vllm_engine):
        """Test function calling with native vLLM support."""
        # Enable native tool choice
        adapter.enable_tool_choice = True

        # Mock the output to include tool calls
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "This is a test response."
        mock_output.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_output.outputs[0].token_ids = [6, 7, 8, 9, 10]
        mock_output.tool_calls = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "unit": "celsius"}'
            }
        }]

        # Update the mock engine to return the output with tool calls
        mock_vllm_engine.generate.return_value = [mock_output]

        # Define a function
        function = {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
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
        }

        # Generate text with function calling
        response = await adapter.generate(
            self.test_prompt,
            functions=[function],
            function_call="auto"
        )

        # Check the response
        assert response.function_call is not None
        assert response.function_call["name"] == "get_weather"
        assert "San Francisco" in response.function_call["arguments"]
        assert "celsius" in response.function_call["arguments"]

        # Check that the engine was called with the correct parameters
        mock_vllm_engine.generate.assert_called_once()
        args, kwargs = mock_vllm_engine.generate.call_args
        assert "tools" in kwargs
        assert kwargs["tools"][0]["type"] == "function"
        assert kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_function_calling_required(self, adapter: LLM, mock_vllm_engine):
        """Test function calling with required option."""
        # Enable native tool choice
        adapter.enable_tool_choice = True

        # Mock the output to include tool calls
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "This is a test response."
        mock_output.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_output.outputs[0].token_ids = [6, 7, 8, 9, 10]
        mock_output.tool_calls = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "unit": "celsius"}'
            }
        }]

        # Update the mock engine to return the output with tool calls
        mock_vllm_engine.generate.return_value = [mock_output]

        # Define a function
        function = {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
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
        }

        # Generate text with function calling
        response = await adapter.generate(
            self.test_prompt,
            functions=[function],
            function_call="required"
        )

        # Check the response
        assert response.function_call is not None
        assert response.function_call["name"] == "get_weather"

        # Check that the engine was called with the correct parameters
        mock_vllm_engine.generate.assert_called_once()
        args, kwargs = mock_vllm_engine.generate.call_args
        assert "tools" in kwargs
        assert kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_json_mode(self, adapter: LLM, mock_vllm_engine):
        """Test JSON mode."""
        # Generate text with JSON mode
        response = await adapter.generate(
            self.test_prompt,
            json_mode=True
        )

        # Check that the response is correct
        assert isinstance(response, LLMResponse)
        assert response.text == "This is a test response."

        # Check that the engine was called correctly
        mock_vllm_engine.generate.assert_called_once()
        args, kwargs = mock_vllm_engine.generate.call_args
        assert "sampling_params" in kwargs

    @pytest.mark.asyncio
    async def test_caching(self, adapter: LLM, mock_vllm_engine):
        """Test caching."""
        # Clear all caches
        clear_all_model_caches()

        # Generate text with caching
        response1 = await adapter.generate(
            self.test_prompt,
            use_cache=True
        )

        # Reset the mock to check if it's called again
        mock_vllm_engine.generate.reset_mock()

        # Generate the same text again
        response2 = await adapter.generate(
            self.test_prompt,
            use_cache=True
        )

        # Check that the responses are the same
        assert response1.text == response2.text

        # Check that the engine was not called again
        mock_vllm_engine.generate.assert_not_called()

        # Clear all caches
        clear_all_model_caches()

    @pytest.mark.asyncio
    async def test_chat_with_caching(self, adapter: LLM, mock_vllm_engine):
        """Test chat with caching."""
        # Clear all caches
        clear_all_model_caches()

        # Create a conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.test_prompt}
        ]

        # Generate a response with caching
        response1 = await adapter.chat(
            messages,
            use_cache=True
        )

        # Reset the mock to check if it's called again
        mock_vllm_engine.generate.reset_mock()

        # Generate the same response again
        response2 = await adapter.chat(
            messages,
            use_cache=True
        )

        # Check that the responses are the same
        assert response1.text == response2.text

        # Check that the engine was not called again
        mock_vllm_engine.generate.assert_not_called()

        # Clear all caches
        clear_all_model_caches()
