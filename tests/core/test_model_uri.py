"""
Tests for the ModelURI class.

This module provides tests for the ModelURI class in Saplings.
"""

import pytest

from saplings.core.model_adapter import ModelURI


class TestModelURI:
    """Test class for the ModelURI class."""

    def test_parse_basic(self):
        """Test parsing a basic URI."""
        uri = "provider://model"
        model_uri = ModelURI.parse(uri)

        assert model_uri.provider == "provider"
        assert model_uri.model_name == "model"
        assert model_uri.version == "latest"
        assert model_uri.parameters == {}

    def test_parse_with_version(self):
        """Test parsing a URI with a version."""
        uri = "provider://model/version"
        model_uri = ModelURI.parse(uri)

        assert model_uri.provider == "provider"
        assert model_uri.model_name == "model"
        assert model_uri.version == "version"
        assert model_uri.parameters == {}

    def test_parse_with_parameters(self):
        """Test parsing a URI with parameters."""
        uri = "provider://model?param1=value1&param2=value2"
        model_uri = ModelURI.parse(uri)

        assert model_uri.provider == "provider"
        assert model_uri.model_name == "model"
        assert model_uri.version == "latest"
        assert model_uri.parameters == {"param1": "value1", "param2": "value2"}

    def test_parse_with_version_and_parameters(self):
        """Test parsing a URI with a version and parameters."""
        uri = "provider://model/version?param1=value1&param2=value2"
        model_uri = ModelURI.parse(uri)

        assert model_uri.provider == "provider"
        assert model_uri.model_name == "model"
        assert model_uri.version == "version"
        assert model_uri.parameters == {"param1": "value1", "param2": "value2"}

    def test_parse_with_complex_model_name(self):
        """Test parsing a URI with a complex model name."""
        uri = "provider://namespace/model-name"
        model_uri = ModelURI.parse(uri)

        assert model_uri.provider == "provider"
        assert model_uri.model_name == "namespace/model-name"
        assert model_uri.version == "latest"
        assert model_uri.parameters == {}

    def test_parse_with_complex_model_name_and_version(self):
        """Test parsing a URI with a complex model name and version."""
        uri = "provider://namespace/model-name/version"
        model_uri = ModelURI.parse(uri)

        assert model_uri.provider == "provider"
        assert model_uri.model_name == "namespace/model-name"
        assert model_uri.version == "version"
        assert model_uri.parameters == {}

    def test_parse_with_complex_model_name_and_parameters(self):
        """Test parsing a URI with a complex model name and parameters."""
        uri = "provider://namespace/model-name?param1=value1&param2=value2"
        model_uri = ModelURI.parse(uri)

        assert model_uri.provider == "provider"
        assert model_uri.model_name == "namespace/model-name"
        assert model_uri.version == "latest"
        assert model_uri.parameters == {"param1": "value1", "param2": "value2"}

    def test_parse_with_complex_model_name_version_and_parameters(self):
        """Test parsing a URI with a complex model name, version, and parameters."""
        uri = "provider://namespace/model-name/version?param1=value1&param2=value2"
        model_uri = ModelURI.parse(uri)

        assert model_uri.provider == "provider"
        assert model_uri.model_name == "namespace/model-name"
        assert model_uri.version == "version"
        assert model_uri.parameters == {"param1": "value1", "param2": "value2"}

    def test_parse_invalid_uri(self):
        """Test parsing an invalid URI."""
        uri = "invalid-uri"
        with pytest.raises(ValueError, match="Invalid model URI: .* Must contain '://'"):
            ModelURI.parse(uri)

    def test_str_representation(self):
        """Test the string representation of a ModelURI."""
        uri = "provider://model/version?param1=value1&param2=value2"
        model_uri = ModelURI.parse(uri)

        assert str(model_uri) == uri

    def test_equality(self):
        """Test equality of ModelURIs."""
        uri1 = "provider://model/version?param1=value1&param2=value2"
        uri2 = "provider://model/version?param1=value1&param2=value2"
        uri3 = "provider://model/version?param1=value1&param3=value3"

        model_uri1 = ModelURI.parse(uri1)
        model_uri2 = ModelURI.parse(uri2)
        model_uri3 = ModelURI.parse(uri3)

        assert model_uri1 == model_uri2
        assert model_uri1 != model_uri3
        assert model_uri1 != "not-a-model-uri"
