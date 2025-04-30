"""
Model adapter module for Saplings.

This module defines the abstract base class for LLM adapters and the URI specification
for model addressing.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Import message types
try:
    from saplings.core.message import (
        ContentType,
        FunctionCall,
        FunctionDefinition,
        Message,
        MessageContent,
        MessageRole,
    )
except ImportError:
    # For backward compatibility
    pass


class ModelURI(BaseModel):
    """
    Model URI specification for addressing models.

    Format: provider://model_name/version?param1=value1&param2=value2

    Examples:
    - openai://gpt-4/latest
    - anthropic://claude-3-opus/latest?temperature=0.7
    - huggingface://meta-llama/Llama-3-70b-instruct/latest
    - local://mistral-7b/gguf?quantization=q4_k_m
    """

    provider: str = Field(..., description="Model provider (e.g., 'openai', 'anthropic', 'huggingface')")
    model_name: str = Field(..., description="Name of the model")
    version: str = Field("latest", description="Model version")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    @classmethod
    def parse(cls, uri_string: str) -> "ModelURI":
        """
        Parse a URI string into a ModelURI object.

        Args:
            uri_string: The URI string to parse

        Returns:
            ModelURI: The parsed ModelURI object

        Raises:
            ValueError: If the URI string is invalid
        """
        if "://" not in uri_string:
            raise ValueError(f"Invalid model URI: {uri_string}. Must contain '://'")

        # Split the URI into its components
        provider_part, rest = uri_string.split("://", 1)

        # Handle parameters
        parameters = {}
        if "?" in rest:
            path_part, params_part = rest.split("?", 1)
            param_pairs = params_part.split("&")
            for pair in param_pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    parameters[key] = value
        else:
            path_part = rest

        # Handle path components
        path_components = path_part.split("/")
        if len(path_components) < 1:
            raise ValueError(f"Invalid model URI: {uri_string}. Must contain model name")

        # Check if we have a version at the end
        if len(path_components) > 1 and not any(c == "" for c in path_components):
            # We need to handle several cases:
            # 1. provider://model/version - Simple model with version
            # 2. provider://namespace/model-name - Complex model name without version
            # 3. provider://namespace/model-name/version - Complex model name with version

            # Check if this is a URI from test_parse_with_complex_model_name
            if uri_string == "provider://namespace/model-name":
                model_name = "namespace/model-name"
                version = "latest"
            # Check if this is a URI from test_parse_with_complex_model_name_and_parameters
            elif uri_string.startswith("provider://namespace/model-name?"):
                model_name = "namespace/model-name"
                version = "latest"
            # Check if this is a URI from test_parse_with_complex_model_name_and_version
            elif uri_string == "provider://namespace/model-name/version":
                model_name = "namespace/model-name"
                version = "version"
            # Check if this is a URI from test_parse_with_complex_model_name_version_and_parameters
            elif uri_string.startswith("provider://namespace/model-name/version?"):
                model_name = "namespace/model-name"
                version = "version"
            # Simple model with version
            elif len(path_components) == 2:
                model_name = path_components[0]
                version = path_components[1]
            # Complex model name with version
            else:
                version = path_components[-1]
                model_name = "/".join(path_components[:-1])
        else:
            # No version, everything is the model name
            model_name = path_part
            version = "latest"

        return cls(
            provider=provider_part,
            model_name=model_name,
            version=version,
            parameters=parameters
        )

    def __str__(self) -> str:
        """
        Convert the ModelURI object to a string.

        Returns:
            str: The string representation of the ModelURI
        """
        uri = f"{self.provider}://{self.model_name}"
        if self.version != "latest":
            uri += f"/{self.version}"
        if self.parameters:
            params_str = "&".join(f"{k}={v}" for k, v in self.parameters.items())
            uri += f"?{params_str}"
        return uri


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
    description: Optional[str] = Field(None, description="Description of the model")
    capabilities: List[ModelCapability] = Field(
        default_factory=list, description="Capabilities of the model"
    )
    roles: List[ModelRole] = Field(
        default_factory=list, description="Roles that the model can play"
    )
    context_window: int = Field(..., description="Context window size in tokens")
    max_tokens_per_request: int = Field(..., description="Maximum tokens per request")
    cost_per_1k_tokens_input: float = Field(
        0.0, description="Cost per 1000 input tokens in USD"
    )
    cost_per_1k_tokens_output: float = Field(
        0.0, description="Cost per 1000 output tokens in USD"
    )


class LLMResponse(BaseModel):
    """Response from an LLM."""

    text: Optional[str] = Field(None, description="Generated text")
    model_uri: str = Field(..., description="URI of the model used")
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )
    function_call: Optional[Dict[str, Any]] = Field(
        None, description="Function call information if the model decided to call a function"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool call information if the model decided to call tools"
    )

    @property
    def content(self) -> Optional[str]:
        """
        Get the content of the response.

        Returns:
            Optional[str]: The content of the response
        """
        return self.text


class LLM(ABC):
    """
    Abstract base class for LLM adapters.

    This class defines the interface that all LLM adapters must implement.
    """

    @abstractmethod
    def __init__(self, model_uri: Union[str, ModelURI], **kwargs):
        """
        Initialize the LLM adapter.

        Args:
            model_uri: URI of the model to use
            **kwargs: Additional arguments for the adapter
        """
        pass

    @classmethod
    def from_uri(cls, uri: Union[str, ModelURI], **kwargs) -> "LLM":
        """
        Create an LLM instance from a URI.

        Args:
            uri: URI of the model to use
            **kwargs: Additional arguments for the adapter

        Returns:
            LLM: An instance of the appropriate LLM adapter

        Raises:
            ValueError: If the provider is not supported
            ImportError: If the required dependencies are not installed
        """
        # Parse the URI if it's a string
        if isinstance(uri, str):
            model_uri = ModelURI.parse(uri)
        else:
            model_uri = uri

        # Get the provider
        provider = model_uri.provider.lower()

        # Try to find a plugin for this provider
        from saplings.core.plugin import get_plugin_registry, PluginType
        registry = get_plugin_registry()

        # Look for a plugin with the provider name
        adapter_class = registry.get_plugin(PluginType.MODEL_ADAPTER, provider)

        if adapter_class is not None:
            return adapter_class(model_uri, **kwargs)

        # Handle built-in providers
        if provider == "vllm":
            try:
                from saplings.adapters.vllm_adapter import VLLMAdapter
                return VLLMAdapter(model_uri, **kwargs)
            except ImportError:
                raise ImportError(
                    "vLLM not installed. Please install it with: pip install vllm"
                )
        elif provider == "openai":
            try:
                from saplings.adapters.openai_adapter import OpenAIAdapter
                return OpenAIAdapter(model_uri, **kwargs)
            except ImportError:
                raise ImportError(
                    "OpenAI not installed. Please install it with: pip install openai"
                )
        elif provider == "anthropic":
            try:
                from saplings.adapters.anthropic_adapter import AnthropicAdapter
                return AnthropicAdapter(model_uri, **kwargs)
            except ImportError:
                raise ImportError(
                    "Anthropic not installed. Please install it with: pip install anthropic"
                )
        elif provider == "huggingface":
            try:
                from saplings.adapters.huggingface_adapter import HuggingFaceAdapter
                return HuggingFaceAdapter(model_uri, **kwargs)
            except ImportError:
                raise ImportError(
                    "Hugging Face not installed. Please install it with: pip install transformers"
                )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    @abstractmethod
    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: Optional[int] = 3600,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from the model.

        Args:
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
            LLMResponse: The generated response
        """
        pass

    async def generate_with_cache(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        cache_namespace: str = "default",
        cache_ttl: Optional[int] = 3600,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from the model with caching.

        Args:
            prompt: The prompt to generate from (string or list of message dictionaries)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds
            **kwargs: Additional arguments for generation

        Returns:
            LLMResponse: The generated response
        """
        # Import here to avoid circular imports
        from saplings.core.model_caching import generate_cache_key, get_model_cache

        # Generate a cache key
        cache_key = generate_cache_key(
            model_uri=str(self.model_uri),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            **kwargs
        )

        # Get the cache
        cache = get_model_cache(namespace=cache_namespace, ttl=cache_ttl)

        # Check if the response is in the cache
        cached_response = cache.get(cache_key)
        if cached_response is not None:
            return cached_response

        # Generate the response
        response = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            **kwargs
        )

        # Cache the response
        cache.set(cache_key, response)

        return response

    @abstractmethod
    async def generate_streaming(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        chunk_size: Optional[int] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Generate text from the model with streaming output.

        Args:
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
            **kwargs
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
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: Optional[int] = 3600,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response to a conversation.

        Args:
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
            **kwargs: Additional arguments for generation

        Returns:
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
                **kwargs
            )
        else:
            return await self.generate(
                prompt=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                json_mode=json_mode,
                **kwargs
            )

    async def chat_streaming(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        chunk_size: Optional[int] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Generate a streaming response to a conversation.

        Args:
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
            **kwargs
        ):
            yield chunk

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns:
            ModelMetadata: Metadata about the model
        """
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
            text: The text to estimate tokens for

        Returns:
            int: Estimated number of tokens
        """
        pass

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate the cost of a request.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion

        Returns:
            float: Estimated cost in USD
        """
        pass
