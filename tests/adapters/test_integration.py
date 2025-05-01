"""
Integration tests for model adapters.

This module provides integration tests for the model adapters in Saplings.
"""

from unittest.mock import MagicMock, patch

import pytest

from saplings.core.model_adapter import LLM, ModelURI
from saplings.core.plugin import PluginType


class TestAdapterIntegration:
    """Integration tests for model adapters."""

    @pytest.mark.asyncio
    async def test_adapter_registration(self, monkeypatch):
        """Test that adapters are registered correctly."""
        # Mock the plugin registry
        mock_registry = MagicMock()
        mock_registry.get_plugin.return_value = None

        with patch("saplings.core.plugin.get_plugin_registry", return_value=mock_registry):
            # Mock the adapter classes
            with patch("saplings.adapters.vllm_adapter.VLLMAdapter", autospec=True) as mock_vllm:
                with patch(
                    "saplings.adapters.openai_adapter.OpenAIAdapter", autospec=True
                ) as mock_openai:
                    with patch(
                        "saplings.adapters.anthropic_adapter.AnthropicAdapter", autospec=True
                    ) as mock_anthropic:
                        with patch(
                            "saplings.adapters.huggingface_adapter.HuggingFaceAdapter",
                            autospec=True,
                        ) as mock_huggingface:
                            # Create models from URIs
                            model1 = LLM.from_uri("vllm://model1")
                            model2 = LLM.from_uri("openai://model2")
                            model3 = LLM.from_uri("anthropic://model3")
                            model4 = LLM.from_uri("huggingface://model4")

                            # Check that the adapters were created correctly
                            mock_vllm.assert_called_once()
                            mock_openai.assert_called_once()
                            mock_anthropic.assert_called_once()
                            mock_huggingface.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_plugin_discovery(self, monkeypatch):
        """Test that adapter plugins are discovered correctly."""
        # Create a mock adapter class
        mock_adapter_class = MagicMock()
        mock_adapter_instance = MagicMock()
        mock_adapter_class.return_value = mock_adapter_instance

        # Mock the plugin registry
        mock_registry = MagicMock()
        mock_registry.get_plugin.return_value = mock_adapter_class

        with patch("saplings.core.plugin.get_plugin_registry", return_value=mock_registry):
            # Create a model from a URI
            model = LLM.from_uri("custom://model")

            # Check that the plugin registry was queried
            mock_registry.get_plugin.assert_called_once()
            args, kwargs = mock_registry.get_plugin.call_args
            assert args[0] == PluginType.MODEL_ADAPTER
            assert args[1] == "custom"

            # Check that the adapter was created correctly
            mock_adapter_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_error_handling(self, monkeypatch):
        """Test error handling in adapters."""
        # Mock the plugin registry
        mock_registry = MagicMock()
        mock_registry.get_plugin.return_value = None

        with patch("saplings.core.plugin.get_plugin_registry", return_value=mock_registry):
            # Mock the import errors
            with patch(
                "saplings.adapters.vllm_adapter.VLLMAdapter",
                side_effect=ImportError("vLLM not installed"),
            ):
                with patch(
                    "saplings.adapters.openai_adapter.OpenAIAdapter",
                    side_effect=ImportError("OpenAI not installed"),
                ):
                    with patch(
                        "saplings.adapters.anthropic_adapter.AnthropicAdapter",
                        side_effect=ImportError("Anthropic not installed"),
                    ):
                        with patch(
                            "saplings.adapters.huggingface_adapter.HuggingFaceAdapter",
                            side_effect=ImportError("HuggingFace not installed"),
                        ):
                            # Check that the correct error is raised
                            with pytest.raises(ImportError, match="vLLM not installed"):
                                LLM.from_uri("vllm://model")

                            with pytest.raises(ImportError, match="OpenAI not installed"):
                                LLM.from_uri("openai://model")

                            with pytest.raises(ImportError, match="Anthropic not installed"):
                                LLM.from_uri("anthropic://model")

                            with pytest.raises(ImportError, match="Hugging Face not installed"):
                                LLM.from_uri("huggingface://model")

    @pytest.mark.asyncio
    async def test_adapter_with_parameters(self, monkeypatch):
        """Test creating adapters with parameters."""
        # Mock the plugin registry
        mock_registry = MagicMock()
        mock_registry.get_plugin.return_value = None

        with patch("saplings.core.plugin.get_plugin_registry", return_value=mock_registry):
            # Mock the adapter classes
            with patch("saplings.adapters.vllm_adapter.VLLMAdapter", autospec=True) as mock_vllm:
                # Create a model from a URI with parameters
                model = LLM.from_uri("vllm://model?temperature=0.5&max_tokens=100")

                # Check that the adapter was created with the correct parameters
                mock_vllm.assert_called_once()
                args, kwargs = mock_vllm.call_args
                assert isinstance(args[0], ModelURI)
                assert args[0].parameters["temperature"] == "0.5"
                assert args[0].parameters["max_tokens"] == "100"
