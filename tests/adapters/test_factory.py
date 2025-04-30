"""
Tests for the LLM factory method.

This module provides tests for the LLM.from_uri factory method.
"""

import pytest
from unittest.mock import MagicMock, patch

from saplings.core.model_adapter import LLM
from saplings.core.plugin import PluginType

# Create mock adapter classes
class MockVLLMAdapter(LLM):
    def __init__(self, model_uri, **kwargs):
        self.model_uri = model_uri
        self.kwargs = kwargs

    async def generate(self, *_, **__):
        pass

    async def generate_streaming(self, *_, **__):
        pass

    def get_metadata(self):
        pass

    def estimate_tokens(self, _):
        pass

    def estimate_cost(self, _, __):
        pass

class MockOpenAIAdapter(MockVLLMAdapter):
    pass

class MockAnthropicAdapter(MockVLLMAdapter):
    pass

class MockHuggingFaceAdapter(MockVLLMAdapter):
    pass

# Create mock modules
mock_vllm_module = MagicMock()
mock_vllm_module.VLLMAdapter = MockVLLMAdapter

mock_openai_module = MagicMock()
mock_openai_module.OpenAIAdapter = MockOpenAIAdapter

mock_anthropic_module = MagicMock()
mock_anthropic_module.AnthropicAdapter = MockAnthropicAdapter

mock_huggingface_module = MagicMock()
mock_huggingface_module.HuggingFaceAdapter = MockHuggingFaceAdapter

# Mock the plugin registry
mock_registry = MagicMock()
mock_registry.get_plugin.return_value = None


class TestLLMFactory:
    """Test class for the LLM factory method."""

    @pytest.mark.asyncio
    async def test_from_uri_vllm(self, monkeypatch):
        """Test creating a vLLM adapter from a URI."""
        # Mock the import system
        def mock_import(name, *args, **kwargs):
            if name == "saplings.adapters.vllm_adapter":
                module = MagicMock()
                module.VLLMAdapter = MockVLLMAdapter
                return module
            return original_import(name, *args, **kwargs)

        original_import = __import__
        monkeypatch.setattr("builtins.__import__", mock_import)

        # Create a model from a URI
        model = LLM.from_uri("vllm://meta-llama/Llama-2-7b-chat-hf")

        # Check that the adapter was created correctly
        assert isinstance(model, MockVLLMAdapter)
        assert model.model_uri.provider == "vllm"
        assert model.model_uri.model_name == "meta-llama"
        assert model.model_uri.version == "Llama-2-7b-chat-hf"

    @pytest.mark.asyncio
    async def test_from_uri_openai(self, monkeypatch):
        """Test creating an OpenAI adapter from a URI."""
        # Mock the import system
        def mock_import(name, *args, **kwargs):
            if name == "saplings.adapters.openai_adapter":
                module = MagicMock()
                module.OpenAIAdapter = MockOpenAIAdapter
                return module
            return original_import(name, *args, **kwargs)

        original_import = __import__
        monkeypatch.setattr("builtins.__import__", mock_import)

        # Create a model from a URI
        model = LLM.from_uri("openai://gpt-4")

        # Check that the adapter was created correctly
        assert isinstance(model, MockOpenAIAdapter)
        assert model.model_uri.provider == "openai"
        assert model.model_uri.model_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_from_uri_anthropic(self, monkeypatch):
        """Test creating an Anthropic adapter from a URI."""
        # Mock the import system
        def mock_import(name, *args, **kwargs):
            if name == "saplings.adapters.anthropic_adapter":
                module = MagicMock()
                module.AnthropicAdapter = MockAnthropicAdapter
                return module
            return original_import(name, *args, **kwargs)

        original_import = __import__
        monkeypatch.setattr("builtins.__import__", mock_import)

        # Create a model from a URI
        model = LLM.from_uri("anthropic://claude-3-opus-20240229")

        # Check that the adapter was created correctly
        assert isinstance(model, MockAnthropicAdapter)
        assert model.model_uri.provider == "anthropic"
        assert model.model_uri.model_name == "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_from_uri_huggingface(self, monkeypatch):
        """Test creating a HuggingFace adapter from a URI."""
        # Mock the import system
        def mock_import(name, *args, **kwargs):
            if name == "saplings.adapters.huggingface_adapter":
                module = MagicMock()
                module.HuggingFaceAdapter = MockHuggingFaceAdapter
                return module
            return original_import(name, *args, **kwargs)

        original_import = __import__
        monkeypatch.setattr("builtins.__import__", mock_import)

        # Create a model from a URI
        model = LLM.from_uri("huggingface://meta-llama/Llama-3-8b-instruct")

        # Check that the adapter was created correctly
        assert isinstance(model, MockHuggingFaceAdapter)
        assert model.model_uri.provider == "huggingface"
        assert model.model_uri.model_name == "meta-llama"
        assert model.model_uri.version == "Llama-3-8b-instruct"

    @pytest.mark.asyncio
    async def test_from_uri_with_parameters(self, monkeypatch):
        """Test creating an adapter from a URI with parameters."""
        # Mock the import system
        def mock_import(name, *args, **kwargs):
            if name == "saplings.adapters.vllm_adapter":
                module = MagicMock()
                module.VLLMAdapter = MockVLLMAdapter
                return module
            return original_import(name, *args, **kwargs)

        original_import = __import__
        monkeypatch.setattr("builtins.__import__", mock_import)

        # Create a model from a URI with parameters
        model = LLM.from_uri("vllm://meta-llama/Llama-2-7b-chat-hf?temperature=0.5&max_tokens=100")

        # Check that the adapter was created correctly
        assert isinstance(model, MockVLLMAdapter)
        assert model.model_uri.provider == "vllm"
        assert model.model_uri.model_name == "meta-llama"
        assert model.model_uri.version == "Llama-2-7b-chat-hf"
        assert model.model_uri.parameters["temperature"] == "0.5"
        assert model.model_uri.parameters["max_tokens"] == "100"

    @pytest.mark.asyncio
    async def test_from_uri_with_version(self, monkeypatch):
        """Test creating an adapter from a URI with a version."""
        # Mock the import system
        def mock_import(name, *args, **kwargs):
            if name == "saplings.adapters.openai_adapter":
                module = MagicMock()
                module.OpenAIAdapter = MockOpenAIAdapter
                return module
            return original_import(name, *args, **kwargs)

        original_import = __import__
        monkeypatch.setattr("builtins.__import__", mock_import)

        # Create a model from a URI with a version
        model = LLM.from_uri("openai://gpt-4/latest")

        # Check that the adapter was created correctly
        assert isinstance(model, MockOpenAIAdapter)
        assert model.model_uri.provider == "openai"
        assert model.model_uri.model_name == "gpt-4"
        assert model.model_uri.version == "latest"

    @pytest.mark.asyncio
    async def test_from_uri_unsupported_provider(self):
        """Test creating an adapter from a URI with an unsupported provider."""
        # Mock the plugin registry to return None
        mock_registry = MagicMock()
        mock_registry.get_plugin.return_value = None

        # Patch the get_plugin_registry function and imports
        with patch("saplings.core.plugin.get_plugin_registry", return_value=mock_registry):
            # Check that an exception is raised
            with pytest.raises(ValueError, match="Unsupported model provider: unsupported"):
                LLM.from_uri("unsupported://model")

    @pytest.mark.asyncio
    async def test_from_uri_plugin(self):
        """Test creating an adapter from a URI using a plugin."""
        # Create a mock adapter class
        class MockCustomAdapter(MockVLLMAdapter):
            pass

        # Mock the plugin registry
        mock_registry = MagicMock()
        mock_registry.get_plugin.return_value = MockCustomAdapter

        # Patch the get_plugin_registry function
        with patch("saplings.core.plugin.get_plugin_registry", return_value=mock_registry):
            # Create a model from a URI
            model = LLM.from_uri("custom://model")

            # Check that the plugin registry was queried
            mock_registry.get_plugin.assert_called_once()
            args = mock_registry.get_plugin.call_args[0]
            assert args[0] == PluginType.MODEL_ADAPTER
            assert args[1] == "custom"

            # Check that the adapter was created
            assert isinstance(model, MockCustomAdapter)
            assert model.model_uri.provider == "custom"
            assert model.model_uri.model_name == "model"
