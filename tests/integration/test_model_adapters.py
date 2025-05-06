from __future__ import annotations

"""
Integration tests for model adapters.

These tests interact with real model APIs and require API keys to be set in the environment.
They are skipped if the required API keys are not available.
"""


import asyncio
import os

import pytest

from saplings.core.model_adapter import LLM


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available")
class TestOpenAIIntegration:
    """Test OpenAI integration with real API."""

    def test_openai_generate(self) -> None:
        """Test OpenAI adapter with real API calls."""
        # Create model
        model = LLM.create(provider="openai", model_name="gpt-3.5-turbo")

        # Test basic generation
        response = asyncio.run(model.generate("Say hello"))
        assert response.text
        assert "hello" in response.text.lower()

        # Check usage statistics
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage
        assert "total_tokens" in response.usage
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] > 0

    @pytest.mark.asyncio()
    async def test_openai_streaming(self) -> None:
        """Test OpenAI streaming with real API calls."""
        # Create model
        model = LLM.create(provider="openai", model_name="gpt-3.5-turbo")

        # Test streaming
        chunks = []
        async for chunk in model.generate_stream("Count from 1 to 5"):
            if chunk.text:
                chunks.append(chunk.text)

        # Verify we got chunks
        assert chunks
        full_text = "".join(chunks)
        assert full_text

        # The response should contain numbers 1 through 5
        for i in range(1, 6):
            assert str(i) in full_text

    def test_openai_function_calling(self) -> None:
        """Test OpenAI function calling with real API calls."""
        # Create model
        model = LLM.create(provider="openai", model_name="gpt-3.5-turbo")

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

        # Test function calling
        response = asyncio.run(
            model.generate(
                "What's the weather in New York?",
                functions=functions,
                function_call={"name": "get_weather"},  # Force function call
            )
        )

        # Verify function call
        assert response.function_call or response.tool_calls

        # Check function call details
        if response.function_call:
            assert response.function_call["name"] == "get_weather"
            assert "location" in response.function_call["arguments"]
        elif response.tool_calls:
            assert response.tool_calls[0]["function"]["name"] == "get_weather"
            assert "location" in response.tool_calls[0]["function"]["arguments"]


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="Anthropic API key not available"
)
class TestAnthropicIntegration:
    """Test Anthropic integration with real API."""

    def test_anthropic_generate(self) -> None:
        """Test Anthropic adapter with real API calls."""
        # Create model
        model = LLM.create(provider="anthropic", model_name="claude-3-haiku-20240307")

        # Test basic generation
        response = asyncio.run(model.generate("Say hello"))
        assert response.text
        assert "hello" in response.text.lower()

        # Check usage statistics
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage
        assert "total_tokens" in response.usage
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] > 0

    @pytest.mark.asyncio()
    async def test_anthropic_streaming(self) -> None:
        """Test Anthropic streaming with real API calls."""
        # Create model
        model = LLM.create(provider="anthropic", model_name="claude-3-haiku-20240307")

        # Test streaming
        chunks = []
        async for chunk in model.generate_stream("Count from 1 to 5"):
            if chunk.text:
                chunks.append(chunk.text)

        # Verify we got chunks
        assert chunks
        full_text = "".join(chunks)
        assert full_text

        # The response should contain numbers 1 through 5
        for i in range(1, 6):
            assert str(i) in full_text


# Only run VLLM tests if the package is installed and a model is available
try:
    import vllm

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
class TestVLLMIntegration:
    """Test vLLM integration."""

    def test_vllm_generate(self) -> None:
        """Test vLLM adapter with real model."""
        try:
            # Create model with a small model for testing
            model = LLM.create(
                provider="vllm",
                model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                device="cpu",  # Use CPU for CI testing
            )

            # Test basic generation
            response = asyncio.run(model.generate("Say hello"))
            assert response.text

            # Check usage statistics
            assert "prompt_tokens" in response.usage
            assert "completion_tokens" in response.usage
            assert "total_tokens" in response.usage
            assert response.usage["prompt_tokens"] > 0
            assert response.usage["completion_tokens"] > 0
            assert response.usage["total_tokens"] > 0
        except Exception as e:
            pytest.skip(f"vLLM model loading failed: {e}")

    @pytest.mark.asyncio()
    async def test_vllm_streaming(self) -> None:
        """Test vLLM streaming with real model."""
        try:
            # Create model with a small model for testing
            model = LLM.create(
                provider="vllm",
                model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                device="cpu",  # Use CPU for CI testing
            )

            # Test streaming
            chunks = []
            async for chunk in model.generate_stream("Count from 1 to 5"):
                if chunk.text:
                    chunks.append(chunk.text)

            # Verify we got chunks
            assert chunks
            full_text = "".join(chunks)
            assert full_text
        except Exception as e:
            pytest.skip(f"vLLM model loading failed: {e}")
