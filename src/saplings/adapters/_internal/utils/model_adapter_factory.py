from __future__ import annotations

"""
Model adapter factory for Saplings.

This module provides a factory for creating model adapters with lazy initialization.
"""

import logging
import weakref
from typing import Any, TypeVar

# Import adapter classes
from saplings.adapters._internal.base import LazyInitializable

# Import lightweight adapters directly
from saplings.adapters._internal.providers import (
    AnthropicAdapter,
    OpenAIAdapter,
)
from saplings.core._internal.exceptions import ModelError
from saplings.core._internal.model_interface import LLM


# Lazy import heavy adapters
def _get_heavy_adapter(name: str):
    """Get heavy adapter using lazy import."""
    from saplings.adapters._internal.providers import __getattr__ as providers_getattr

    return providers_getattr(name)


logger = logging.getLogger(__name__)

# Type variable for model adapter classes
T = TypeVar("T", bound=LLM)

# Dictionary to store model instances by key
_model_instances: dict[str, weakref.ReferenceType[Any]] = {}


class ModelAdapterFactory:
    """
    Factory for creating model adapters with lazy initialization.

    This factory implements the factory pattern for creating model adapters,
    with support for lazy initialization and dependency injection.
    """

    @staticmethod
    def create_adapter(
        provider: str,
        model_name: str,
        lazy_init: bool = True,
        **kwargs: Any,
    ) -> LLM:
        """
        Create a model adapter instance.

        Args:
        ----
            provider: The model provider (e.g., 'vllm', 'openai', 'anthropic')
            model_name: The model name
            lazy_init: Whether to initialize the model lazily
            **kwargs: Additional parameters for the model

        Returns:
        -------
            LLM: An instance of the appropriate model adapter

        Raises:
        ------
            ValueError: If the provider is not supported
            ModelError: If model initialization fails

        """
        # Create a cache key
        cache_key = ModelAdapterFactory._create_cache_key(provider, model_name, **kwargs)

        # Check if the model already exists in the cache
        if cache_key in _model_instances:
            existing_model = _model_instances[cache_key]()
            if existing_model is not None and isinstance(existing_model, LLM):
                logger.debug(f"Using existing model instance from cache: {cache_key}")
                # If the existing model is LazyInitializable and not initialized,
                # initialize it if lazy_init is False
                if (
                    not lazy_init
                    and isinstance(existing_model, LazyInitializable)
                    and not existing_model.is_initialized
                ):
                    existing_model.initialize()
                return existing_model

        # Create the adapter based on the provider
        try:
            # Pass lazy_init to the adapter constructor if it accepts it
            adapter_kwargs = kwargs.copy()
            adapter_kwargs["_lazy_init"] = lazy_init  # Use underscore prefix to avoid conflicts

            # Create the adapter instance
            adapter = ModelAdapterFactory._create_adapter_instance(
                provider, model_name, **adapter_kwargs
            )

            # Store in cache
            _model_instances[cache_key] = weakref.ref(adapter)

            # If lazy initialization is disabled, initialize the model now
            if not lazy_init:
                ModelAdapterFactory._initialize_adapter(adapter)

            return adapter
        except Exception as e:
            # Wrap the original exception to provide context
            msg = f"Failed to create model adapter: {e}"
            raise ModelError(msg, cause=e)

    @staticmethod
    def _create_cache_key(provider: str, model_name: str, **kwargs: Any) -> str:
        """
        Create a cache key for the model adapter.

        Args:
        ----
            provider: The model provider
            model_name: The model name
            **kwargs: Additional parameters

        Returns:
        -------
            str: Cache key

        """
        cache_key = f"{provider}:{model_name}"
        if kwargs:
            params_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key += f":{params_str}"
        return cache_key

    @staticmethod
    def _create_adapter_instance(provider: str, model_name: str, **kwargs: Any) -> LLM:
        """
        Create an instance of the appropriate model adapter.

        Args:
        ----
            provider: The model provider
            model_name: The model name
            **kwargs: Additional parameters

        Returns:
        -------
            LLM: Model adapter instance

        Raises:
        ------
            ValueError: If the provider is not supported

        """
        provider_lower = provider.lower()

        if provider_lower == "openai":
            return OpenAIAdapter(provider, model_name, **kwargs)
        elif provider_lower == "anthropic":
            return AnthropicAdapter(provider, model_name, **kwargs)
        elif provider_lower == "vllm":
            try:
                VLLMAdapter = _get_heavy_adapter("VLLMAdapter")
                return VLLMAdapter(provider, model_name, **kwargs)
            except (ImportError, RuntimeError):
                logger.warning(
                    "VLLMAdapter initialization failed. Falling back to VLLMFallbackAdapter."
                )
                VLLMFallbackAdapter = _get_heavy_adapter("VLLMFallbackAdapter")
                return VLLMFallbackAdapter(provider, model_name, **kwargs)
        elif provider_lower == "huggingface":
            HuggingFaceAdapter = _get_heavy_adapter("HuggingFaceAdapter")
            return HuggingFaceAdapter(provider, model_name, **kwargs)
        elif provider_lower == "transformers":
            TransformersAdapter = _get_heavy_adapter("TransformersAdapter")
            return TransformersAdapter(provider, model_name, **kwargs)
        else:
            msg = f"Unsupported model provider: {provider}"
            raise ValueError(msg)

    @staticmethod
    def _initialize_adapter(adapter: LLM) -> None:
        """
        Initialize a model adapter.

        This method is called when lazy initialization is disabled.

        Args:
        ----
            adapter: The model adapter to initialize

        """
        # Check if the adapter implements LazyInitializable
        if isinstance(adapter, LazyInitializable):
            # Use the LazyInitializable interface
            adapter.initialize()
        # Fallback for adapters that don't implement LazyInitializable
        elif hasattr(adapter, "initialize") and callable(adapter.initialize):
            # Call the initialize method directly
            adapter.initialize()
