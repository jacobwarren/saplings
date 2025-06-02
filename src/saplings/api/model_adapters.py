from __future__ import annotations

"""
Model adapters API module for Saplings.

This module provides the public API for model adapters.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from saplings.api.stability import beta, stable


# Model Capability
@stable
class ModelCapability(str, Enum):
    """
    Capabilities of a language model.

    This enum defines the possible capabilities of a language model.
    """

    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function_calling"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    AUDIO = "audio"
    CODE = "code"
    REASONING = "reasoning"
    CUSTOM = "custom"


# Model Role
@stable
class ModelRole(str, Enum):
    """
    Roles of a language model.

    This enum defines the possible roles of a language model.
    """

    GENERAL = "general"
    PLANNER = "planner"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    JUDGE = "judge"
    RETRIEVER = "retriever"
    SUMMARIZER = "summarizer"
    CUSTOM = "custom"


# Model Metadata
@stable
class ModelMetadata:
    """
    Metadata for a language model.

    This class encapsulates metadata about a language model, including
    its name, provider, and capabilities.
    """

    def __init__(
        self,
        name: str,
        provider: str,
        capabilities: Optional[List[ModelCapability]] = None,
        roles: Optional[List[ModelRole]] = None,
        context_window: int = 4096,
        token_limit: int = 4096,
        cost_per_token: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model metadata.

        Args:
        ----
            name: Name of the model
            provider: Provider of the model
            capabilities: Capabilities of the model
            roles: Roles the model can fulfill
            context_window: Context window size in tokens
            token_limit: Maximum number of tokens per request
            cost_per_token: Cost per token in USD
            metadata: Additional metadata

        """
        self.name = name
        self.provider = provider
        self.capabilities = capabilities or []
        self.roles = roles or []
        self.context_window = context_window
        self.token_limit = token_limit
        self.cost_per_token = cost_per_token
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of metadata

        """
        return {
            "name": self.name,
            "provider": self.provider,
            "capabilities": [str(c) for c in self.capabilities],
            "roles": [str(r) for r in self.roles],
            "context_window": self.context_window,
            "token_limit": self.token_limit,
            "cost_per_token": self.cost_per_token,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """
        Create metadata from a dictionary.

        Args:
        ----
            data: Dictionary representation of metadata

        Returns:
        -------
            ModelMetadata: Model metadata

        """
        capabilities = [ModelCapability(c) for c in data.get("capabilities", [])]
        roles = [ModelRole(r) for r in data.get("roles", [])]

        return cls(
            name=data.get("name", "unknown"),
            provider=data.get("provider", "unknown"),
            capabilities=capabilities,
            roles=roles,
            context_window=data.get("context_window", 4096),
            token_limit=data.get("token_limit", 4096),
            cost_per_token=data.get("cost_per_token", 0.0),
            metadata=data.get("metadata", {}),
        )


# LLM Response
@stable
class LLMResponse:
    """
    Response from a language model.

    This class encapsulates the response from a language model, including
    the generated text, token usage, and other metadata.
    """

    def __init__(
        self,
        content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model_name: Optional[str] = None,
        finish_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a response.

        Args:
        ----
            content: Generated text
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total number of tokens
            model_name: Name of the model
            finish_reason: Reason for finishing generation
            metadata: Additional metadata

        """
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens or (prompt_tokens + completion_tokens)
        self.model_name = model_name
        self.finish_reason = finish_reason
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to a dictionary.

        Returns
        -------
            Dict[str, Any]: Dictionary representation of response

        """
        return {
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMResponse":
        """
        Create response from a dictionary.

        Args:
        ----
            data: Dictionary representation of response

        Returns:
        -------
            LLMResponse: Model response

        """
        return cls(
            content=data.get("content", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            model_name=data.get("model_name"),
            finish_reason=data.get("finish_reason"),
            metadata=data.get("metadata", {}),
        )


# Base LLM
@stable
class LLM(ABC):
    """
    Base class for language model adapters.

    This class defines the interface for all language model adapters
    in the Saplings framework. It provides methods for generating text,
    embedding text, and managing model resources.
    """

    def __init__(self, model_name: str, provider: str, **kwargs):
        """
        Initialize a language model adapter.

        Args:
        ----
            model_name: Name of the model
            provider: Provider of the model
            **kwargs: Additional configuration options

        """
        self.model_name = model_name
        self.provider = provider
        self.config = kwargs
        self._metadata = None

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Returns:
        -------
            LLMResponse: Generated response

        """

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate text from a prompt with streaming.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Yields:
        ------
            str: Generated text chunks

        """

    @abstractmethod
    async def embed(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Embed text.

        Args:
        ----
            text: Text to embed
            **kwargs: Additional embedding options

        Returns:
        -------
            Union[List[float], List[List[float]]]: Embeddings

        """

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Model metadata

        """

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
        ----
            text: Text to count tokens for

        Returns:
        -------
            int: Number of tokens

        """


# LLM Builder
@stable
class LLMBuilder:
    """
    Builder for creating LLM instances with a fluent interface.

    This builder provides a convenient way to configure and create LLM
    instances with various options and dependencies.
    """

    def __init__(self):
        """Initialize the LLM builder."""
        self._provider = None
        self._model_name = None
        self._config = {}

    def with_provider(self, provider: str) -> "LLMBuilder":
        """
        Set the provider for the LLM.

        Args:
        ----
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
        -------
            Self for method chaining

        """
        self._provider = provider
        return self

    def with_model_name(self, model_name: str) -> "LLMBuilder":
        """
        Set the model name for the LLM.

        Args:
        ----
            model_name: Name of the model to use

        Returns:
        -------
            Self for method chaining

        """
        self._model_name = model_name
        return self

    def with_config(self, config: Dict[str, Any]) -> "LLMBuilder":
        """
        Set the configuration for the LLM.

        Args:
        ----
            config: Configuration dictionary

        Returns:
        -------
            Self for method chaining

        """
        self._config.update(config)
        return self

    def build(self) -> LLM:
        """
        Build the LLM instance.

        Returns
        -------
            LLM: Language model instance

        """
        if not self._provider:
            raise ValueError("Provider is required")
        if not self._model_name:
            raise ValueError("Model name is required")

        provider = self._provider.lower()

        if provider == "openai":
            return OpenAIAdapter(model_name=self._model_name, **self._config)
        elif provider == "anthropic":
            return AnthropicAdapter(model_name=self._model_name, **self._config)
        elif provider == "huggingface":
            return HuggingFaceAdapter(model_name=self._model_name, **self._config)
        elif provider == "vllm":
            return VLLMAdapter(model_name=self._model_name, **self._config)
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")


# OpenAI Adapter
@stable
class OpenAIAdapter(LLM):
    """
    Adapter for OpenAI models.

    This adapter provides an interface for using OpenAI models with
    the Saplings framework.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize an OpenAI adapter.

        Args:
        ----
            model_name: Name of the model
            api_key: OpenAI API key
            organization: OpenAI organization ID
            **kwargs: Additional configuration options

        """
        super().__init__(model_name=model_name, provider="openai", **kwargs)
        self.api_key = api_key
        self.organization = organization
        self._client = None

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Returns:
        -------
            LLMResponse: Generated response

        """
        # Implementation would go here
        return LLMResponse(
            content="This is a placeholder response from OpenAI",
            prompt_tokens=len(prompt.split()),
            completion_tokens=10,
            model_name=self.model_name,
        )

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate text from a prompt with streaming.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Yields:
        ------
            str: Generated text chunks

        """
        # Implementation would go here
        chunks = ["This ", "is ", "a ", "placeholder ", "response ", "from ", "OpenAI"]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.1)

    async def embed(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Embed text.

        Args:
        ----
            text: Text to embed
            **kwargs: Additional embedding options

        Returns:
        -------
            Union[List[float], List[List[float]]]: Embeddings

        """
        # Implementation would go here
        if isinstance(text, str):
            return [0.1] * 10
        return [[0.1] * 10 for _ in text]

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Model metadata

        """
        if self._metadata is None:
            self._metadata = ModelMetadata(
                name=self.model_name,
                provider="openai",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING,
                ],
                roles=[ModelRole.GENERAL],
                context_window=8192,
                token_limit=4096,
                cost_per_token=0.00002,
            )
        return self._metadata

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
        ----
            text: Text to count tokens for

        Returns:
        -------
            int: Number of tokens

        """
        # Implementation would go here
        return len(text.split())


# Anthropic Adapter
@stable
class AnthropicAdapter(LLM):
    """
    Adapter for Anthropic models.

    This adapter provides an interface for using Anthropic models with
    the Saplings framework.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize an Anthropic adapter.

        Args:
        ----
            model_name: Name of the model
            api_key: Anthropic API key
            **kwargs: Additional configuration options

        """
        super().__init__(model_name=model_name, provider="anthropic", **kwargs)
        self.api_key = api_key
        self._client = None

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Returns:
        -------
            LLMResponse: Generated response

        """
        # Implementation would go here
        return LLMResponse(
            content="This is a placeholder response from Anthropic",
            prompt_tokens=len(prompt.split()),
            completion_tokens=10,
            model_name=self.model_name,
        )

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate text from a prompt with streaming.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Yields:
        ------
            str: Generated text chunks

        """
        # Implementation would go here
        chunks = ["This ", "is ", "a ", "placeholder ", "response ", "from ", "Anthropic"]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.1)

    async def embed(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Embed text.

        Args:
        ----
            text: Text to embed
            **kwargs: Additional embedding options

        Returns:
        -------
            Union[List[float], List[List[float]]]: Embeddings

        """
        # Implementation would go here
        if isinstance(text, str):
            return [0.1] * 10
        return [[0.1] * 10 for _ in text]

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Model metadata

        """
        if self._metadata is None:
            self._metadata = ModelMetadata(
                name=self.model_name,
                provider="anthropic",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                ],
                roles=[ModelRole.GENERAL],
                context_window=100000,
                token_limit=4096,
                cost_per_token=0.00001,
            )
        return self._metadata

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
        ----
            text: Text to count tokens for

        Returns:
        -------
            int: Number of tokens

        """
        # Implementation would go here
        return len(text.split())


# HuggingFace Adapter
@beta
class HuggingFaceAdapter(LLM):
    """
    Adapter for Hugging Face models.

    This adapter provides an interface for using Hugging Face models with
    the Saplings framework.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize a Hugging Face adapter.

        Args:
        ----
            model_name: Name of the model
            api_key: Hugging Face API key
            **kwargs: Additional configuration options

        """
        super().__init__(model_name=model_name, provider="huggingface", **kwargs)
        self.api_key = api_key
        self._client = None

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Returns:
        -------
            LLMResponse: Generated response

        """
        # Implementation would go here
        return LLMResponse(
            content="This is a placeholder response from Hugging Face",
            prompt_tokens=len(prompt.split()),
            completion_tokens=10,
            model_name=self.model_name,
        )

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate text from a prompt with streaming.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Yields:
        ------
            str: Generated text chunks

        """
        # Implementation would go here
        chunks = ["This ", "is ", "a ", "placeholder ", "response ", "from ", "Hugging Face"]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.1)

    async def embed(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Embed text.

        Args:
        ----
            text: Text to embed
            **kwargs: Additional embedding options

        Returns:
        -------
            Union[List[float], List[List[float]]]: Embeddings

        """
        # Implementation would go here
        if isinstance(text, str):
            return [0.1] * 10
        return [[0.1] * 10 for _ in text]

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Model metadata

        """
        if self._metadata is None:
            self._metadata = ModelMetadata(
                name=self.model_name,
                provider="huggingface",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                ],
                roles=[ModelRole.GENERAL],
                context_window=2048,
                token_limit=2048,
                cost_per_token=0.0,
            )
        return self._metadata

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
        ----
            text: Text to count tokens for

        Returns:
        -------
            int: Number of tokens

        """
        # Implementation would go here
        return len(text.split())


# VLLM Adapter
@beta
class VLLMAdapter(LLM):
    """
    Adapter for vLLM models.

    This adapter provides an interface for using vLLM models with
    the Saplings framework.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize a vLLM adapter.

        Args:
        ----
            model_name: Name of the model
            **kwargs: Additional configuration options

        """
        super().__init__(model_name=model_name, provider="vllm", **kwargs)
        self._client = None

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Returns:
        -------
            LLMResponse: Generated response

        """
        # Implementation would go here
        return LLMResponse(
            content="This is a placeholder response from vLLM",
            prompt_tokens=len(prompt.split()),
            completion_tokens=10,
            model_name=self.model_name,
        )

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate text from a prompt with streaming.

        Args:
        ----
            prompt: Prompt for generation
            **kwargs: Additional generation options

        Yields:
        ------
            str: Generated text chunks

        """
        # Implementation would go here
        chunks = ["This ", "is ", "a ", "placeholder ", "response ", "from ", "vLLM"]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.1)

    async def embed(
        self, text: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Embed text.

        Args:
        ----
            text: Text to embed
            **kwargs: Additional embedding options

        Returns:
        -------
            Union[List[float], List[List[float]]]: Embeddings

        """
        # Implementation would go here
        if isinstance(text, str):
            return [0.1] * 10
        return [[0.1] * 10 for _ in text]

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Model metadata

        """
        if self._metadata is None:
            self._metadata = ModelMetadata(
                name=self.model_name,
                provider="vllm",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                ],
                roles=[ModelRole.GENERAL],
                context_window=4096,
                token_limit=4096,
                cost_per_token=0.0,
            )
        return self._metadata

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
        ----
            text: Text to count tokens for

        Returns:
        -------
            int: Number of tokens

        """
        # Implementation would go here
        return len(text.split())


__all__ = [
    # Enums
    "ModelCapability",
    "ModelRole",
    # Core classes
    "ModelMetadata",
    "LLMResponse",
    "LLM",
    "LLMBuilder",
    # Adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "VLLMAdapter",
]
