"""
Test proper error handling for missing API keys and configuration.

This module tests Task 10.3: Implement proper error handling for missing API keys and configuration.
Tests comprehensive configuration validation, API key checking, and helpful error messages.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from saplings.api.agent import AgentConfig


class TestAgentConfigErrorHandling:
    """Test comprehensive error handling in AgentConfig."""

    def test_missing_api_key_openai(self):
        """Test that missing OpenAI API key produces helpful error message."""
        # Clear any existing API key
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig(provider="openai", model_name="gpt-4o")
            result = config.validate()

            assert not result.is_valid
            assert "OpenAI API key" in result.message or "Openai API key" in result.message
            assert "OPENAI_API_KEY" in str(result.suggestions)
            assert "environment variable" in str(result.suggestions)
            assert "api_key parameter" in str(result.suggestions)

    def test_missing_api_key_anthropic(self):
        """Test that missing Anthropic API key produces helpful error message."""
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig(provider="anthropic", model_name="claude-3-opus")
            result = config.validate()

            assert not result.is_valid
            assert "Anthropic API key" in result.message or "anthropic API key" in result.message
            assert "ANTHROPIC_API_KEY" in str(result.suggestions)

    def test_api_key_from_environment(self):
        """Test that API key is correctly detected from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            config = AgentConfig(provider="openai", model_name="gpt-4o")
            result = config.validate()
            assert result.is_valid

    def test_api_key_from_parameter(self):
        """Test that API key passed as parameter works."""
        config = AgentConfig(provider="openai", model_name="gpt-4o", api_key="test-key-123")
        result = config.validate()
        assert result.is_valid

    def test_invalid_model_name_helpful_error(self):
        """Test that invalid model names produce helpful error messages."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = AgentConfig(provider="openai", model_name="invalid-model")
            result = config.validate()

            assert not result.is_valid
            assert "invalid-model" in result.message
            assert "Valid models" in str(result.suggestions) or "valid models" in str(
                result.suggestions
            )
            assert len(result.suggestions) > 0

    def test_missing_dependencies_helpful_error(self):
        """Test that missing dependencies produce helpful error messages."""
        config = AgentConfig(provider="huggingface", model_name="gpt2")
        result = config.check_dependencies()

        # Should identify missing transformers dependency
        if not result.all_available:
            assert "transformers" in result.message
            assert "pip install" in result.message

    def test_configuration_suggestions(self):
        """Test that configuration provides actionable suggestions."""
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig(provider="openai", model_name="gpt-4o")
            suggestions = config.suggest_fixes()

            assert len(suggestions) > 0
            assert any("OPENAI_API_KEY" in suggestion for suggestion in suggestions)
            assert any("export" in suggestion for suggestion in suggestions)

    def test_configuration_explanation(self):
        """Test that configuration can explain its settings."""
        config = AgentConfig(
            provider="openai", model_name="gpt-4o", enable_gasa=True, enable_monitoring=False
        )
        explanation = config.explain()

        assert "provider" in explanation
        assert "openai" in explanation
        assert "GASA" in explanation
        assert "Monitoring" in explanation

    def test_provider_specific_validation(self):
        """Test provider-specific validation rules."""
        # Test local provider doesn't require API key
        config = AgentConfig(provider="test", model_name="test-model")
        result = config.validate()
        assert result.is_valid

        # Test vLLM provider validation
        config = AgentConfig(provider="vllm", model_name="Qwen/Qwen3-7B-Instruct")
        result = config.check_dependencies()
        # Should check for vLLM installation

    def test_comprehensive_validation_workflow(self):
        """Test complete validation workflow."""
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig(provider="openai", model_name="gpt-4o")

            # Full validation should catch multiple issues
            result = config.validate()
            assert not result.is_valid

            # Should provide help URL
            assert result.help_url is not None
            assert "openai" in result.help_url

            # Should provide multiple suggestions
            assert len(result.suggestions) >= 2

    def test_error_message_quality_standards(self):
        """Test that error messages meet quality standards."""
        with patch.dict(os.environ, {}, clear=True):
            config = AgentConfig(provider="openai", model_name="invalid-model")
            result = config.validate()

            # Error message should be specific and actionable
            assert not result.is_valid
            assert len(result.message) > 20  # Not too brief
            assert len(result.message) < 500  # Not too verbose
            assert "." in result.message or result.message.endswith("'")  # Proper punctuation
            assert result.message[0].isupper()  # Proper capitalization


# The validation methods are now implemented in the actual AgentConfig class
