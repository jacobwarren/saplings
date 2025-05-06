from __future__ import annotations

"""
Unit tests for the VLLM adapter.
"""


import asyncio
from unittest.mock import MagicMock, patch

import pytest

from saplings.adapters.vllm_adapter import VLLMAdapter

# Skip all tests if vLLM is not installed
pytestmark = pytest.mark.skipif(
    not hasattr(VLLMAdapter, "VLLM_AVAILABLE") or not VLLMAdapter.VLLM_AVAILABLE,
    reason="vLLM not installed",
)


class TestVLLMAdapter:
    """Test the VLLM adapter."""

    def test_vllm_adapter_initialization(self) -> None:
        """Test VLLM adapter initialization with various configurations."""
        # Test with basic configuration
        with patch("vllm.LLM") as mock_llm:
            adapter = VLLMAdapter(provider="vllm", model_name="Qwen/Qwen3-7B-Instruct")
            assert adapter.model_name == "Qwen/Qwen3-7B-Instruct"
            assert adapter.provider == "vllm"
            mock_llm.assert_called_once()

        # Test with quantization
        with patch("vllm.LLM") as mock_llm:
            adapter = VLLMAdapter(
                provider="vllm", model_name="Qwen/Qwen3-7B-Instruct", quantization="awq"
            )
            assert adapter.quantization == "awq"
            mock_llm.assert_called_once()
            # Verify quantization was passed correctly
            assert mock_llm.call_args[1]["quantization"] == "awq"

    def test_vllm_adapter_generate(self) -> None:
        """Test VLLM adapter generate method."""
        # Create mock engine
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.outputs = [
            MagicMock(text="Hello, world!", token_ids=[1, 2, 3, 4, 5], finished=True)
        ]
        mock_engine.generate.return_value = mock_response

        # Create adapter with mock engine
        adapter = VLLMAdapter(provider="vllm", model_name="Qwen/Qwen3-7B-Instruct")
        adapter.engine = mock_engine

        # Mock the tokenizer to return a fixed number of tokens
        adapter.get_token_count = MagicMock(return_value=10)

        # Test generate method
        response = asyncio.run(adapter.generate("Say hello"))
        assert response.text == "Hello, world!"
        assert response.provider == "vllm"
        assert response.model_name == "Qwen/Qwen3-7B-Instruct"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5
        assert response.usage["total_tokens"] == 15
        mock_engine.generate.assert_called_once()

    def test_vllm_adapter_function_calling(self) -> None:
        """Test VLLM adapter function calling."""
        # Skip if function calling is not supported
        try:
            import vllm

            if not hasattr(vllm, "SamplingParams") or not hasattr(
                vllm.SamplingParams, "tool_choice"
            ):
                pytest.skip("Function calling not supported in this vLLM version")
        except (ImportError, AttributeError):
            pytest.skip("Function calling not supported in this vLLM version")

        # Create mock engine
        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.outputs = [
            MagicMock(
                text='{"function": "get_weather", "arguments": {"location": "New York"}}',
                token_ids=[1, 2, 3, 4, 5],
                finished=True,
            )
        ]
        mock_engine.generate.return_value = mock_response

        # Create adapter with mock engine
        adapter = VLLMAdapter(
            provider="vllm", model_name="meta-llama/Llama-3.1-8B-Instruct", enable_tool_choice=True
        )
        adapter.engine = mock_engine

        # Mock the tokenizer to return a fixed number of tokens
        adapter.get_token_count = MagicMock(return_value=10)

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

        # The function call should be parsed from the JSON response
        assert response.function_call or response.tool_calls
        mock_engine.generate.assert_called_once()

    @pytest.mark.asyncio()
    async def test_vllm_adapter_streaming(self) -> None:
        """Test VLLM adapter streaming."""
        # Create mock engine and streaming response
        mock_engine = MagicMock()

        # Create mock RequestOutput objects for streaming
        mock_output1 = MagicMock(text="Hello", token_ids=[1, 2], finished=False)
        mock_output2 = MagicMock(text="Hello, ", token_ids=[1, 2, 3], finished=False)
        mock_output3 = MagicMock(text="Hello, world!", token_ids=[1, 2, 3, 4, 5], finished=True)

        # Set up the mock to return different outputs on each call
        mock_engine.generate.side_effect = [
            MagicMock(outputs=[mock_output1]),
            MagicMock(outputs=[mock_output2]),
            MagicMock(outputs=[mock_output3]),
        ]

        # Create adapter with mock engine
        adapter = VLLMAdapter(provider="vllm", model_name="Qwen/Qwen3-7B-Instruct")
        adapter.engine = mock_engine

        # Mock the tokenizer to return a fixed number of tokens
        adapter.get_token_count = MagicMock(return_value=5)

        # Test streaming
        chunks = []
        async for chunk in adapter.generate_stream("Say hello"):
            if chunk.text:
                chunks.append(chunk.text)

        # We should get the differences between outputs
        assert chunks == ["Hello", ", ", "world!"]
        assert mock_engine.generate.call_count == 3
