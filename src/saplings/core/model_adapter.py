from __future__ import annotations

"""
Model adapter module for Saplings.

This module provides a unified interface for interacting with different LLM providers.
It includes a model registry to ensure only one instance of a model with the same
configuration is created, which helps reduce memory usage and improve performance.
"""


import logging
import weakref
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

# Type variable for LLM subclasses
T = TypeVar("T", bound="LLM")

# Flag to enable model registry
ENABLE_MODEL_REGISTRY = True

# Dictionary to store model instances by URI if needed
_model_instances: dict[str, weakref.ReferenceType[Any]] = {}

# Import message types


class ModelRole(str, Enum):
    """Roles that a model can play in the system."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    JUDGE = "judge"
    VALIDATOR = "validator"
    GENERAL = "general"


class ModelCapability(str, Enum):
    """Capabilities that a model can have."""

    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    STRUCTURED_OUTPUT = "structured_output"
    JSON_MODE = "json_mode"


class ModelMetadata(BaseModel):
    """Metadata about a model."""

    name: str = Field(..., description="Name of the model")
    provider: str = Field(..., description="Provider of the model")
    version: str = Field(..., description="Version of the model")
    description: str | None = Field(None, description="Description of the model")
    capabilities: list[ModelCapability] = Field(
        default_factory=list, description="Capabilities of the model"
    )
    roles: list[ModelRole] = Field(
        default_factory=list, description="Roles that the model can play"
    )
    context_window: int = Field(..., description="Context window size in tokens")
    max_tokens_per_request: int = Field(..., description="Maximum tokens per request")
    cost_per_1k_tokens_input: float = Field(0.0, description="Cost per 1000 input tokens in USD")
    cost_per_1k_tokens_output: float = Field(0.0, description="Cost per 1000 output tokens in USD")


class LLMResponse(BaseModel):
    """Response from an LLM."""

    text: str | None = Field(None, description="Generated text")
    provider: str = Field(..., description="Provider of the model")
    model_name: str = Field(..., description="Name of the model")
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )
    function_call: dict[str, Any] | None = Field(
        None, description="Function call information if the model decided to call a function"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        None, description="Tool call information if the model decided to call tools"
    )

    @property
    def content(self):
        """
        Get the content of the response.

        Returns
        -------
            Optional[str]: The content of the response

        """
        return self.text


class LLM(ABC):
    """
    Abstract base class for LLM adapters.

    This class defines the interface that all LLM adapters must implement.
    """

    provider: str
    model_name: str

    @abstractmethod
    def __init__(self, provider: str, model_name: str, **kwargs) -> None:
        """
        Initialize the LLM adapter.

        Args:
        ----
            provider: The model provider (e.g., 'vllm', 'openai', 'anthropic')
            model_name: The model name
            **kwargs: Additional arguments for the adapter

        """
        self.provider = provider
        self.model_name = model_name

    @classmethod
    def create(cls, provider: str, model_name: str, **kwargs) -> "LLM":
        """
        Create an LLM instance with a provider and model name.

        Args:
        ----
            provider: The model provider (e.g., 'vllm', 'openai', 'anthropic', 'huggingface')
            model_name: The model name
            **kwargs: Additional parameters for the model

        Returns:
        -------
            LLM: An instance of the appropriate LLM adapter

        Raises:
        ------
            ValueError: If the provider is not supported
            ImportError: If the required dependencies are not installed

        """
        # Import here to avoid circular imports
        from saplings.core.model_registry import ModelRegistry, create_model_key, get_model_registry

        # Create a model key for the registry
        model_key = create_model_key(provider, model_name, **kwargs)

        # Check if model registry is enabled
        if ENABLE_MODEL_REGISTRY:
            try:
                # Check if the model already exists in the registry
                model_registry = get_model_registry()
                # Ensure we have a valid ModelRegistry instance
                if model_registry is not None and isinstance(model_registry, ModelRegistry):
                    existing_model = model_registry.get(model_key)
                    if existing_model is not None:
                        # Return the existing model instance
                        logger.debug(
                            f"Using existing model instance from registry: {provider}/{model_name}"
                        )
                        return existing_model
                else:
                    logger.warning(
                        "Model registry not available or invalid type. Using fallback cache."
                    )
            except Exception as e:
                logger.warning(f"Error accessing model registry: {e}. Using fallback cache.")
        else:
            # Create a cache key string
            cache_key = f"{provider}:{model_name}"
            if kwargs:
                params_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key += f":{params_str}"

            # Use the simple dictionary-based cache as fallback
            if cache_key in _model_instances:
                existing_model = _model_instances[cache_key]()
                if existing_model is not None:
                    logger.debug(f"Using existing model instance from cache: {cache_key}")
                    return existing_model

        # Try to find a plugin for this provider
        from saplings.core.model_registry import ModelRegistry, get_model_registry
        from saplings.core.plugin import PluginType, get_plugin_registry

        plugin_registry = get_plugin_registry()
        try:
            model_registry = get_model_registry()
            if not isinstance(model_registry, ModelRegistry):
                logger.warning(
                    "Model registry is not a valid ModelRegistry instance. Creating a new one."
                )
                model_registry = ModelRegistry()
        except Exception as e:
            logger.warning(f"Error getting model registry: {e}. Creating a new one.")
            model_registry = ModelRegistry()

        # Look for a plugin with the provider name
        adapter_class = plugin_registry.get_plugin(PluginType.MODEL_ADAPTER, provider.lower())

        if adapter_class is not None:
            try:
                # Create the adapter instance
                # Check if the adapter class is a ModelAdapterPlugin
                from saplings.core.plugin import ModelAdapterPlugin

                # Create the plugin instance
                plugin_instance = adapter_class()

                # Check if it's a ModelAdapterPlugin
                if isinstance(plugin_instance, ModelAdapterPlugin):
                    # Use the create_adapter method
                    model_instance = plugin_instance.create_adapter(provider, model_name, **kwargs)
                else:
                    # Try to instantiate directly
                    model_instance = adapter_class()

                # Verify it's an LLM instance
                if not isinstance(model_instance, LLM):
                    logger.warning(
                        f"Plugin {adapter_class.__name__} does not return an LLM instance. Skipping."
                    )
                    # Continue to built-in providers
                else:
                    # Register the model instance in the registry
                    if (
                        ENABLE_MODEL_REGISTRY
                        and model_registry is not None
                        and isinstance(model_registry, ModelRegistry)
                    ):
                        try:
                            model_registry.register(model_key, model_instance)
                        except Exception as e:
                            logger.warning(
                                f"Error registering model in registry: {e}. Using fallback cache."
                            )
                            _model_instances[cache_key] = weakref.ref(model_instance)
                    else:
                        # Use the simple dictionary-based cache as fallback
                        _model_instances[cache_key] = weakref.ref(model_instance)

                    return model_instance
            except Exception as e:
                logger.warning(
                    f"Failed to create model instance from plugin {adapter_class.__name__}: {e}"
                )
                # Continue to built-in providers

        # Handle built-in providers
        if provider.lower() == "vllm":
            # First check if vLLM is available at the module level
            try:
                import vllm

                # Just try to import it to check if it's available
                vllm_available = True
                # Log the vLLM version if available
                logger.debug(f"vLLM version: {getattr(vllm, '__version__', 'unknown')}")
            except ImportError:
                vllm_available = False

            if not vllm_available:
                # Try to use the fallback adapter if available
                try:
                    from saplings.adapters.vllm_fallback_adapter import VLLMFallbackAdapter

                    # Log the fallback
                    logger.warning("vLLM not installed. Falling back to VLLMFallbackAdapter.")

                    # Create the fallback adapter
                    model_instance = VLLMFallbackAdapter(provider, model_name, **kwargs)

                    # Register in registry or cache
                    if (
                        ENABLE_MODEL_REGISTRY
                        and model_registry is not None
                        and isinstance(model_registry, ModelRegistry)
                    ):
                        try:
                            model_registry.register(model_key, model_instance)
                        except Exception as e:
                            logger.warning(
                                f"Error registering model in registry: {e}. Using fallback cache."
                            )
                            _model_instances[cache_key] = weakref.ref(model_instance)
                    else:
                        _model_instances[cache_key] = weakref.ref(model_instance)

                    return model_instance
                except ImportError:
                    # If fallback adapter is not available, raise the original error
                    msg = "vLLM not installed. Please install it with: pip install vllm"
                    raise ImportError(msg)

            # If vLLM is available, try to create the adapter
            try:
                from saplings.adapters.vllm_adapter import VLLMAdapter

                try:
                    # Try to create the VLLMAdapter
                    model_instance = VLLMAdapter(provider, model_name, **kwargs)

                    # Register the model instance in the registry
                    if (
                        ENABLE_MODEL_REGISTRY
                        and model_registry is not None
                        and isinstance(model_registry, ModelRegistry)
                    ):
                        try:
                            model_registry.register(model_key, model_instance)
                        except Exception as e:
                            logger.warning(
                                f"Error registering model in registry: {e}. Using fallback cache."
                            )
                            _model_instances[cache_key] = weakref.ref(model_instance)
                    else:
                        # Use the simple dictionary-based cache as fallback
                        _model_instances[cache_key] = weakref.ref(model_instance)

                    return model_instance
                except RuntimeError as e:
                    # Check if the error is related to Triton
                    if "triton" in str(e).lower() or "failed to be inspected" in str(e).lower():
                        # Import the fallback adapter
                        from saplings.adapters.vllm_fallback_adapter import VLLMFallbackAdapter

                        # Log the fallback
                        logger.warning(
                            f"vLLM initialization failed due to Triton issues: {e}. "
                            f"Falling back to VLLMFallbackAdapter."
                        )

                        # Create the fallback adapter
                        model_instance = VLLMFallbackAdapter(provider, model_name, **kwargs)

                        # Register the model instance in the registry
                        if (
                            ENABLE_MODEL_REGISTRY
                            and model_registry is not None
                            and isinstance(model_registry, ModelRegistry)
                        ):
                            try:
                                model_registry.register(model_key, model_instance)
                            except Exception as e:
                                logger.warning(
                                    f"Error registering model in registry: {e}. Using fallback cache."
                                )
                                _model_instances[cache_key] = weakref.ref(model_instance)
                        else:
                            # Use the simple dictionary-based cache as fallback
                            _model_instances[cache_key] = weakref.ref(model_instance)

                        return model_instance
                    # Re-raise other errors
                    raise
            except ImportError as e:
                # This should not happen since we already checked if vLLM is available
                # But just in case, try to use the fallback adapter
                try:
                    from saplings.adapters.vllm_fallback_adapter import VLLMFallbackAdapter

                    # Log the fallback
                    logger.warning(
                        f"Error importing VLLMAdapter: {e}. Falling back to VLLMFallbackAdapter."
                    )

                    # Create the fallback adapter
                    model_instance = VLLMFallbackAdapter(provider, model_name, **kwargs)

                    # Register in registry or cache
                    if (
                        ENABLE_MODEL_REGISTRY
                        and model_registry is not None
                        and isinstance(model_registry, ModelRegistry)
                    ):
                        try:
                            model_registry.register(model_key, model_instance)
                        except Exception as e:
                            logger.warning(
                                f"Error registering model in registry: {e}. Using fallback cache."
                            )
                            _model_instances[cache_key] = weakref.ref(model_instance)
                    else:
                        _model_instances[cache_key] = weakref.ref(model_instance)

                    return model_instance
                except ImportError:
                    # If fallback adapter is not available, raise the original error
                    msg = "vLLM not installed. Please install it with: pip install vllm"
                    raise ImportError(msg)
        elif provider.lower() == "openai":
            try:
                from saplings.adapters.openai_adapter import OpenAIAdapter

                # Create the adapter
                model_instance = OpenAIAdapter(provider, model_name, **kwargs)

                # Register the model instance in the registry
                if (
                    ENABLE_MODEL_REGISTRY
                    and model_registry is not None
                    and isinstance(model_registry, ModelRegistry)
                ):
                    try:
                        model_registry.register(model_key, model_instance)
                    except Exception as e:
                        logger.warning(
                            f"Error registering model in registry: {e}. Using fallback cache."
                        )
                        _model_instances[cache_key] = weakref.ref(model_instance)
                else:
                    # Use the simple dictionary-based cache as fallback
                    _model_instances[cache_key] = weakref.ref(model_instance)

                return model_instance
            except ImportError:
                msg = "OpenAI not installed. Please install it with: pip install openai"
                raise ImportError(msg)
        elif provider.lower() == "anthropic":
            try:
                from saplings.adapters.anthropic_adapter import AnthropicAdapter

                # Create the adapter
                model_instance = AnthropicAdapter(provider, model_name, **kwargs)

                # Register the model instance in the registry
                if (
                    ENABLE_MODEL_REGISTRY
                    and model_registry is not None
                    and isinstance(model_registry, ModelRegistry)
                ):
                    try:
                        model_registry.register(model_key, model_instance)
                    except Exception as e:
                        logger.warning(
                            f"Error registering model in registry: {e}. Using fallback cache."
                        )
                        _model_instances[cache_key] = weakref.ref(model_instance)
                else:
                    # Use the simple dictionary-based cache as fallback
                    _model_instances[cache_key] = weakref.ref(model_instance)

                return model_instance
            except ImportError:
                msg = "Anthropic not installed. Please install it with: pip install anthropic"
                raise ImportError(msg)
        elif provider.lower() == "huggingface":
            try:
                from saplings.adapters.huggingface_adapter import HuggingFaceAdapter

                # Create the adapter
                model_instance = HuggingFaceAdapter(provider, model_name, **kwargs)

                # Register the model instance in the registry
                if (
                    ENABLE_MODEL_REGISTRY
                    and model_registry is not None
                    and isinstance(model_registry, ModelRegistry)
                ):
                    try:
                        model_registry.register(model_key, model_instance)
                    except Exception as e:
                        logger.warning(
                            f"Error registering model in registry: {e}. Using fallback cache."
                        )
                        _model_instances[cache_key] = weakref.ref(model_instance)
                else:
                    # Use the simple dictionary-based cache as fallback
                    _model_instances[cache_key] = weakref.ref(model_instance)

                return model_instance
            except ImportError:
                msg = "Hugging Face not installed. Please install it with: pip install transformers"
                raise ImportError(msg)
        elif provider.lower() == "transformers":
            try:
                from saplings.adapters.transformers_adapter import TransformersAdapter

                # Create the adapter
                model_instance = TransformersAdapter(provider, model_name, **kwargs)

                # Register the model instance in the registry
                if (
                    ENABLE_MODEL_REGISTRY
                    and model_registry is not None
                    and isinstance(model_registry, ModelRegistry)
                ):
                    try:
                        model_registry.register(model_key, model_instance)
                    except Exception as e:
                        logger.warning(
                            f"Error registering model in registry: {e}. Using fallback cache."
                        )
                        _model_instances[cache_key] = weakref.ref(model_instance)
                else:
                    # Use the simple dictionary-based cache as fallback
                    _model_instances[cache_key] = weakref.ref(model_instance)

                return model_instance
            except ImportError:
                msg = "Transformers not installed. Please install it with: pip install transformers"
                raise ImportError(msg)
        else:
            msg = f"Unsupported model provider: {provider}"
            raise ValueError(msg)

    # Removed from_uri method as part of the model_uri removal

    @abstractmethod
    async def generate(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: int | None = 3600,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from the model.

        Args:
        ----
            prompt: The prompt to generate from (string or list of message dictionaries)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            use_cache: Whether to use caching for this request
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds
            **kwargs: Additional arguments for generation

        Returns:
        -------
            LLMResponse: The generated response

        """

    async def generate_with_cache(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
        cache_namespace: str = "model",
        cache_ttl: int | None = 3600,
        cache_provider: str = "memory",
        cache_strategy: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from the model with caching.

        Args:
        ----
            prompt: The prompt to generate from (string or list of message dictionaries)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds
            cache_provider: Cache provider to use
            cache_strategy: Cache eviction strategy
            **kwargs: Additional arguments for generation

        Returns:
        -------
            LLMResponse: The generated response

        """
        # Import here to avoid circular imports
        from saplings.core.caching import generate_with_cache_async
        from saplings.core.caching.interface import CacheStrategy

        # Convert strategy string to enum if provided
        strategy = CacheStrategy.LRU  # Default to LRU
        if cache_strategy:
            strategy = CacheStrategy(cache_strategy)

        # Define the generate function
        async def _generate(p, **kw):
            return await self.generate(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                json_mode=json_mode,
                **kw,
            )

        # Create cache key based on provider and model_name
        cache_key = f"{self.provider}:{self.model_name}"

        # Use the unified caching system
        return await generate_with_cache_async(
            generate_func=_generate,
            cache_key=cache_key,  # Use provider:model_name format as the cache key
            prompt=prompt,
            namespace=cache_namespace,
            ttl=cache_ttl,
            provider=cache_provider,
            strategy=strategy,
            **kwargs,
        )

    @abstractmethod
    async def generate_streaming(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        chunk_size: int | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
        **kwargs,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """
        Generate text from the model with streaming output.

        Args:
        ----
            prompt: The prompt to generate from (string or list of message dictionaries)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            chunk_size: Number of tokens per chunk (if supported by the provider)
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            **kwargs: Additional arguments for generation

        Yields:
        ------
            Union[str, Dict[str, Any]]: Text chunks or function call chunks as they are generated

        """
        # Default implementation that falls back to non-streaming generate
        # Subclasses should override this with native streaming if available
        response = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            **kwargs,
        )
        if response.text:
            yield response.text
        elif response.function_call:
            yield {"function_call": response.function_call}
        elif response.tool_calls:
            for tool_call in response.tool_calls:
                yield {"tool_call": tool_call}

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "model",
        cache_ttl: int | None = 3600,
        cache_provider: str = "memory",
        cache_strategy: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response to a conversation.

        Args:
        ----
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            use_cache: Whether to use caching for this request
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds
            cache_provider: Cache provider to use
            cache_strategy: Cache eviction strategy
            **kwargs: Additional arguments for generation

        Returns:
        -------
            LLMResponse: The generated response

        """
        if use_cache:
            return await self.generate_with_cache(
                prompt=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                json_mode=json_mode,
                cache_namespace=cache_namespace,
                cache_ttl=cache_ttl,
                cache_provider=cache_provider,
                cache_strategy=cache_strategy,
                **kwargs,
            )
        return await self.generate(
            prompt=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            **kwargs,
        )

    async def chat_streaming(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        chunk_size: int | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, str] | None = None,
        json_mode: bool = False,
        **kwargs,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """
        Generate a streaming response to a conversation.

        Args:
        ----
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            chunk_size: Number of tokens per chunk (if supported by the provider)
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            **kwargs: Additional arguments for generation

        Yields:
        ------
            Union[str, Dict[str, Any]]: Text chunks or function call chunks as they are generated

        """
        async for chunk in self.generate_streaming(
            prompt=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            chunk_size=chunk_size,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            **kwargs,
        ):
            yield chunk

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Metadata about the model

        """

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
        ----
            text: The text to estimate tokens for

        Returns:
        -------
            int: Estimated number of tokens

        """

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate the cost of a request.

        Args:
        ----
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion

        Returns:
        -------
            float: Estimated cost in USD

        """
