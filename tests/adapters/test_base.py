"""
Base test class for model adapters.

This module provides a base test class with common test cases for all model adapters.
"""

import asyncio
import pytest
from typing import Any, Dict, Optional, Type, Union

from saplings.core.model_adapter import LLM, LLMResponse, ModelURI


class BaseAdapterTest:
    """Base test class for model adapters."""
    
    # The adapter class to test
    adapter_class: Type[LLM] = None
    
    # The provider name for the adapter
    provider_name: str = None
    
    # The model name to use for testing
    model_name: str = "test-model"
    
    # Additional parameters for the adapter
    adapter_kwargs: Dict[str, Any] = {}
    
    # Test prompt to use
    test_prompt: str = "This is a test prompt."
    
    @pytest.fixture
    def model_uri(self) -> str:
        """Create a model URI for testing."""
        assert self.provider_name is not None, "Provider name must be set"
        return f"{self.provider_name}://{self.model_name}"
    
    @pytest.fixture
    def adapter(self, model_uri: str) -> LLM:
        """Create an adapter instance for testing."""
        assert self.adapter_class is not None, "Adapter class must be set"
        return self.adapter_class(model_uri, **self.adapter_kwargs)
    
    @pytest.mark.asyncio
    async def test_initialization(self, adapter: LLM):
        """Test that the adapter initializes correctly."""
        assert adapter is not None
        assert isinstance(adapter, LLM)
        assert adapter.model_uri.provider == self.provider_name
        assert adapter.model_uri.model_name == self.model_name
    
    @pytest.mark.asyncio
    async def test_generate(self, adapter: LLM, monkeypatch):
        """Test the generate method."""
        # This should be overridden by subclasses to mock the actual API call
        pass
    
    @pytest.mark.asyncio
    async def test_generate_streaming(self, adapter: LLM, monkeypatch):
        """Test the generate_streaming method."""
        # This should be overridden by subclasses to mock the actual API call
        pass
    
    @pytest.mark.asyncio
    async def test_get_metadata(self, adapter: LLM):
        """Test the get_metadata method."""
        metadata = adapter.get_metadata()
        assert metadata is not None
        assert metadata.name == self.model_name
        assert metadata.provider == self.provider_name
    
    @pytest.mark.asyncio
    async def test_estimate_tokens(self, adapter: LLM):
        """Test the estimate_tokens method."""
        tokens = adapter.estimate_tokens(self.test_prompt)
        assert tokens > 0
    
    @pytest.mark.asyncio
    async def test_estimate_cost(self, adapter: LLM):
        """Test the estimate_cost method."""
        cost = adapter.estimate_cost(100, 100)
        assert cost >= 0
    
    @pytest.mark.asyncio
    async def test_from_uri(self, model_uri: str, monkeypatch):
        """Test creating the adapter from a URI."""
        # Mock the adapter class to avoid actual initialization
        def mock_init(self, model_uri, **kwargs):
            if isinstance(model_uri, str):
                self.model_uri = ModelURI.parse(model_uri)
            else:
                self.model_uri = model_uri
        
        monkeypatch.setattr(self.adapter_class, "__init__", mock_init)
        
        # Test creating the adapter from a URI
        adapter = LLM.from_uri(model_uri)
        assert adapter is not None
        assert isinstance(adapter, self.adapter_class)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, adapter: LLM, monkeypatch):
        """Test error handling in the adapter."""
        # This should be overridden by subclasses to test specific error scenarios
        pass
