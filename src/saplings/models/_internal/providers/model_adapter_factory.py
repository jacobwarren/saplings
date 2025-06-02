from __future__ import annotations

"""
Model adapter factory for Saplings.

This module provides a factory for creating model adapters with lazy initialization.
"""

import importlib
import logging
import weakref
from typing import Any, TypeVar, cast

from saplings.core._internal.exceptions import ModelError
from saplings.models._internal.interfaces import LLM, LazyInitializable

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

    This class implements the ModelAdapterFactory protocol defined in interfaces.py.
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
                if not lazy_init:
                    # Check if the model is LazyInitializable
                    if isinstance(existing_model, LazyInitializable):
                        # Use type casting to satisfy the type checker
                        lazy_model = cast(LazyInitializable, existing_model)
                        if not lazy_model.is_initialized:
                            lazy_model.initialize()
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

        # Import adapter classes dynamically
        adapter_mapping = {
            "openai": "saplings.models._internal.providers.openai_adapter",
            "anthropic": "saplings.models._internal.providers.anthropic_adapter",
            "vllm": "saplings.models._internal.providers.vllm_adapter",
            "huggingface": "saplings.models._internal.providers.huggingface_adapter",
            "transformers": "saplings.models._internal.providers.transformers_adapter",
        }

        if provider_lower in adapter_mapping:
            try:
                module_path = adapter_mapping[provider_lower]
                module = importlib.import_module(module_path)

                # Get the adapter class name
                class_name = f"{provider_lower.capitalize()}Adapter"
                if provider_lower == "openai":
                    class_name = "OpenAIAdapter"
                elif provider_lower == "vllm":
                    class_name = "VLLMAdapter"

                adapter_class = getattr(module, class_name)
                return adapter_class(provider, model_name, **kwargs)
            except (ImportError, AttributeError) as e:
                if provider_lower == "vllm":
                    # Try fallback adapter
                    try:
                        fallback_module = importlib.import_module(
                            "saplings.models._internal.providers.vllm_fallback_adapter"
                        )
                        fallback_class = fallback_module.VLLMFallbackAdapter
                        logger.warning(
                            "VLLMAdapter initialization failed. Falling back to VLLMFallbackAdapter."
                        )
                        return fallback_class(provider, model_name, **kwargs)
                    except (ImportError, AttributeError):
                        pass
                msg = f"Failed to import adapter for provider {provider}: {e}"
                raise ImportError(msg) from e
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
            # Use the LazyInitializable interface with type casting
            lazy_adapter = cast(LazyInitializable, adapter)
            lazy_adapter.initialize()
            return

        # Fallback for adapters that don't implement LazyInitializable
        if hasattr(adapter, "initialize") and callable(adapter.initialize):
            # Call the initialize method directly
            adapter.initialize()
