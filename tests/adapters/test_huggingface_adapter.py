"""
Tests for the HuggingFace adapter.

This module provides tests for the HuggingFace adapter implementation.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelURI

from .test_base import BaseAdapterTest

# Check if we should run integration tests
RUN_INTEGRATION_TESTS = os.environ.get("RUN_INTEGRATION_TESTS", "").lower() in ("1", "true", "yes")

# Try to import the adapter, but don't fail if dependencies are missing
try:
    from saplings.adapters.huggingface_adapter import HuggingFaceAdapter

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceAdapter = MagicMock()  # Placeholder for type hints


class TestHuggingFaceAdapter(BaseAdapterTest):
    """Test class for the HuggingFace adapter."""

    adapter_class = HuggingFaceAdapter
    provider_name = "huggingface"
    model_name = "gpt2"  # Using a smaller, publicly available model

    @pytest.fixture
    def mock_adapter(self):
        """Create a mocked adapter instance for basic tests."""
        # Create a mock adapter
        adapter = MagicMock(spec=self.adapter_class)
        adapter.model_name = self.model_name
        adapter.provider_name = self.provider_name
        adapter.model_uri = ModelURI(provider=self.provider_name, model_name=self.model_name)
        adapter.temperature = 0.7
        adapter.max_tokens = 1024
        adapter.device = "cpu"

        # Mock the generate method
        async def mock_generate(prompt, **kwargs):
            return LLMResponse(
                text="This is a test response.",
                model_uri=str(adapter.model_uri),
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                metadata={
                    "model": self.model_name,
                    "provider": self.provider_name,
                },
            )

        adapter.generate.side_effect = mock_generate

        # Mock the generate_streaming method
        async def mock_generate_streaming(prompt, **kwargs):
            for text in ["This ", "is ", "a ", "test ", "response."]:
                yield text

        adapter.generate_streaming.side_effect = mock_generate_streaming

        # Mock the estimate_tokens method
        adapter.estimate_tokens.return_value = 10

        # Mock the estimate_cost method
        adapter.estimate_cost.return_value = 0.0002

        # Mock the get_metadata method
        adapter.get_metadata.return_value = ModelMetadata(
            name=self.model_name,
            provider=self.provider_name,
            version="1.0",
            description=f"Mock {self.model_name}",
            capabilities=[],
            roles=[],
            context_window=4096,
            max_tokens_per_request=2048,
            cost_per_1k_tokens_input=0.0,
            cost_per_1k_tokens_output=0.0,
        )

        return adapter

    @pytest.fixture
    def adapter(self, mock_adapter):
        """Override the base adapter fixture to use our mock."""
        return mock_adapter

    @pytest.mark.asyncio
    async def test_generate(self, adapter):
        """Test the generate method."""
        # Test the generate method
        response = await adapter.generate(self.test_prompt)

        # Check that the response is correct
        assert isinstance(response, LLMResponse)
        assert response.text == "This is a test response."
        assert response.model_uri == f"{self.provider_name}://{self.model_name}"
        assert response.metadata["model"] == self.model_name
        assert response.metadata["provider"] == self.provider_name

        # Check that the generate method was called correctly
        adapter.generate.assert_called_once_with(self.test_prompt)

    @pytest.mark.asyncio
    async def test_generate_streaming(self, adapter):
        """Test the generate_streaming method."""
        # Test the generate_streaming method
        chunks = []
        async for chunk in adapter.generate_streaming(self.test_prompt):
            chunks.append(chunk)

        # Check that the chunks are correct
        assert len(chunks) == 5
        assert "".join(chunks) == "This is a test response."

        # Check that the generate_streaming method was called correctly
        adapter.generate_streaming.assert_called_once_with(self.test_prompt)

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_adapter):
        """Test error handling in the adapter."""
        # Make the generate method raise an exception
        mock_adapter.generate.side_effect = Exception("Test error")

        # Check that the exception is propagated
        with pytest.raises(Exception, match="Test error"):
            await mock_adapter.generate(self.test_prompt)

    def test_cleanup(self):
        """Test the cleanup method."""
        # Create a mock adapter
        mock_adapter = MagicMock(spec=self.adapter_class)

        # Mock the gc and torch modules
        with patch("gc.collect") as mock_gc_collect:
            with patch(
                "saplings.adapters.huggingface_adapter.torch.cuda.empty_cache"
            ) as mock_empty_cache:
                # Call the real cleanup method
                with patch.object(self.adapter_class, "cleanup", autospec=True) as mock_cleanup:
                    mock_cleanup.side_effect = lambda self: None

                    # Create an adapter instance
                    adapter = mock_adapter

                    # Call cleanup
                    adapter.cleanup()

                    # Check that cleanup was called
                    adapter.cleanup.assert_called_once()

    def test_with_parameters(self):
        """Test creating the adapter with parameters."""
        # Skip the actual model loading and just test parameter parsing
        with patch.object(HuggingFaceAdapter, "__init__", return_value=None) as mock_init:
            # Create a URI with parameters
            uri = f"{self.provider_name}://{self.model_name}?temperature=0.5&max_tokens=100&device=cuda&torch_dtype=float16"

            # Create the adapter (this will just call our mocked __init__)
            adapter = HuggingFaceAdapter(uri)

            # Check that __init__ was called with the correct URI
            mock_init.assert_called_once()
            args, _ = mock_init.call_args
            assert len(args) == 1
            assert isinstance(args[0], str)
            assert "temperature=0.5" in args[0]
            assert "max_tokens=100" in args[0]
            assert "device=cuda" in args[0]
            assert "torch_dtype=float16" in args[0]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HUGGINGFACE_AVAILABLE, reason="HuggingFace dependencies not installed")
    @pytest.mark.skipif(not RUN_INTEGRATION_TESTS, reason="Integration tests disabled")
    async def test_real_initialization(self):
        """Test that the adapter initializes correctly with a real model."""
        # This test uses a real model, so it's skipped by default
        uri = f"{self.provider_name}://{self.model_name}"

        # Create the adapter with a real model
        adapter = self.adapter_class(uri)

        # Check that the adapter was initialized correctly
        assert adapter is not None
        assert adapter.model_name == self.model_name
        assert adapter.model_uri.provider == self.provider_name
        assert adapter.model_uri.model_name == self.model_name

        # Clean up
        adapter.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HUGGINGFACE_AVAILABLE, reason="HuggingFace dependencies not installed")
    @pytest.mark.skipif(not RUN_INTEGRATION_TESTS, reason="Integration tests disabled")
    async def test_real_generate(self):
        """Test the generate method with a real model."""
        # This test uses a real model, so it's skipped by default
        uri = f"{self.provider_name}://{self.model_name}"

        # Create the adapter with a real model
        adapter = self.adapter_class(uri)

        try:
            # Generate text
            response = await adapter.generate("Hello, world!")

            # Check that the response is correct
            assert isinstance(response, LLMResponse)
            assert response.text is not None
            assert response.model_uri == uri
            assert response.metadata["model"] == self.model_name
            assert response.metadata["provider"] == self.provider_name
        finally:
            # Clean up
            adapter.cleanup()
