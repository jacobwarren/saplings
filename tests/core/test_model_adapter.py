"""
Tests for the model_adapter module.
"""

import pytest
from pydantic import ValidationError

from saplings.core.model_adapter import (
    LLMResponse,
    ModelCapability,
    ModelMetadata,
    ModelRole,
    ModelURI,
)


class TestModelURI:
    """Tests for the ModelURI class."""

    def test_parse_valid_uri(self):
        """Test parsing a valid URI."""
        uri_string = "openai://gpt-4/latest?temperature=0.7&top_p=0.9"
        uri = ModelURI.parse(uri_string)

        assert uri.provider == "openai"
        assert uri.model_name == "gpt-4"
        assert uri.version == "latest"
        assert uri.parameters == {"temperature": "0.7", "top_p": "0.9"}

    def test_parse_uri_without_version(self):
        """Test parsing a URI without a version."""
        uri_string = "anthropic://claude-3-opus"
        uri = ModelURI.parse(uri_string)

        assert uri.provider == "anthropic"
        assert uri.model_name == "claude-3-opus"
        assert uri.version == "latest"
        assert uri.parameters == {}

    def test_parse_uri_without_parameters(self):
        """Test parsing a URI without parameters."""
        uri_string = "huggingface://meta-llama/Llama-3-70b-instruct/latest"
        uri = ModelURI.parse(uri_string)

        assert uri.provider == "huggingface"
        assert uri.model_name == "meta-llama/Llama-3-70b-instruct"
        assert uri.version == "latest"
        assert uri.parameters == {}

    def test_parse_invalid_uri(self):
        """Test parsing an invalid URI."""
        uri_string = "invalid-uri"
        with pytest.raises(ValueError):
            ModelURI.parse(uri_string)

    def test_str_representation(self):
        """Test string representation of a ModelURI."""
        uri = ModelURI(
            provider="openai",
            model_name="gpt-4",
            version="latest",
            parameters={"temperature": 0.7, "top_p": 0.9},
        )

        uri_string = str(uri)
        # The order of parameters in the query string is not guaranteed
        assert uri_string.startswith("openai://gpt-4")
        assert "temperature=0.7" in uri_string
        assert "top_p=0.9" in uri_string


class TestModelMetadata:
    """Tests for the ModelMetadata class."""

    def test_valid_metadata(self):
        """Test creating valid model metadata."""
        metadata = ModelMetadata(
            name="GPT-4",
            provider="OpenAI",
            version="latest",
            description="Advanced language model from OpenAI",
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
            roles=[ModelRole.GENERAL, ModelRole.EXECUTOR],
            context_window=8192,
            max_tokens_per_request=4096,
            cost_per_1k_tokens_input=0.01,
            cost_per_1k_tokens_output=0.03,
        )

        assert metadata.name == "GPT-4"
        assert metadata.provider == "OpenAI"
        assert metadata.version == "latest"
        assert metadata.description == "Advanced language model from OpenAI"
        assert ModelCapability.TEXT_GENERATION in metadata.capabilities
        assert ModelCapability.CODE_GENERATION in metadata.capabilities
        assert ModelRole.GENERAL in metadata.roles
        assert ModelRole.EXECUTOR in metadata.roles
        assert metadata.context_window == 8192
        assert metadata.max_tokens_per_request == 4096
        assert metadata.cost_per_1k_tokens_input == 0.01
        assert metadata.cost_per_1k_tokens_output == 0.03

    def test_missing_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            ModelMetadata(
                name="GPT-4",
                provider="OpenAI",
                # Missing version
                context_window=8192,
                max_tokens_per_request=4096,
            )


class TestLLMResponse:
    """Tests for the LLMResponse class."""

    def test_valid_response(self):
        """Test creating a valid LLM response."""
        response = LLMResponse(
            text="This is a generated response.",
            model_uri="openai://gpt-4/latest",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            metadata={"finish_reason": "stop", "latency_ms": 500},
        )

        assert response.text == "This is a generated response."
        assert response.model_uri == "openai://gpt-4/latest"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30
        assert response.metadata["finish_reason"] == "stop"
        assert response.metadata["latency_ms"] == 500

    def test_missing_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            LLMResponse(
                # Missing model_uri
                text="This is a test response."
            )
