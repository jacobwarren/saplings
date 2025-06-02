from __future__ import annotations

"""
Model interfaces module for Saplings.

This module provides the core interfaces for model adapters and LLM components.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Type variable for LLM subclasses
T = TypeVar("T", bound="LLM")


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


@runtime_checkable
class LazyInitializable(Protocol):
    """Protocol for lazy initializable objects."""

    @property
    def is_initialized(self) -> bool:
        """Whether the object is initialized."""
        ...

    def initialize(self) -> None:
        """Initialize the object."""
        ...


class ModelAdapterFactory(Protocol):
    """Protocol for model adapter factory."""

    @staticmethod
    def create_adapter(
        provider: str,
        model_name: str,
        lazy_init: bool = True,
        **kwargs: Any,
    ) -> "LLM":
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

        """
        ...


class LLM(ABC):
    """
    Abstract base class for LLM adapters.

    This class defines the interface that all LLM adapters must implement.
    """

    provider: str
    model_name: str

    # Class variable for the factory
    _factory: ClassVar[ModelAdapterFactory | None] = None

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

    @classmethod
    def create(cls, provider: str, model_name: str, **kwargs: Any) -> "LLM":
        """
        Create a model adapter instance.

        Args:
        ----
            provider: The model provider (e.g., 'vllm', 'openai', 'anthropic')
            model_name: The model name
            **kwargs: Additional parameters for the model

        Returns:
        -------
            LLM: An instance of the appropriate model adapter

        """
        if cls._factory is None:
            raise RuntimeError("Model adapter factory not set. Call LLM.set_factory() first.")

        return cls._factory.create_adapter(provider, model_name, **kwargs)

    @classmethod
    def set_factory(cls, factory: ModelAdapterFactory) -> None:
        """
        Set the model adapter factory.

        Args:
        ----
            factory: The model adapter factory

        """
        cls._factory = factory
