from __future__ import annotations

"""
Interface for model caching service.

This module defines the interface for the model caching service.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Union

from saplings.core._internal.model_interface import LLM, LLMResponse


@dataclass
class ModelCachingConfig:
    """Configuration for model caching operations."""

    cache_provider: str = "memory"
    ttl: int = 3600
    namespace: str = "model"
    strategy: str = "lru"
    max_size: Optional[int] = None


class IModelCachingService(Protocol):
    """Interface for model caching service."""

    @abstractmethod
    async def get_model(self, timeout: Optional[float] = None) -> LLM:
        """
        Get the LLM instance.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            LLM: The initialized model

        Raises:
        ------
            ModelError: If the model is not initialized
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        ...

    @abstractmethod
    async def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, timeout: Optional[float] = None
    ) -> float:
        """
        Estimate cost for token usage.

        Args:
        ----
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            timeout: Optional timeout in seconds

        Returns:
        -------
            float: Estimated cost in USD

        Raises:
        ------
            ModelError: If cost estimation fails
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        ...

    @abstractmethod
    async def generate_text_with_cache(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from the model with caching.

        Args:
        ----
            prompt: The prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            timeout: Optional timeout in seconds
            **kwargs: Additional arguments for generation

        Returns:
        -------
            str: The generated text

        Raises:
        ------
            ModelError: If generation fails
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        ...

    @abstractmethod
    async def generate_response_with_cache(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a full response from the model with caching.

        Args:
        ----
            prompt: The prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            timeout: Optional timeout in seconds
            **kwargs: Additional arguments for generation

        Returns:
        -------
            LLMResponse: The complete model response

        Raises:
        ------
            ModelError: If generation fails
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        ...

    @abstractmethod
    async def get_cached_response(
        self,
        prompt: str,
        model_name: str,
        config: Optional[ModelCachingConfig] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a prompt and model.

        Args:
        ----
            prompt: The prompt to check in cache
            model_name: The model name
            config: Optional caching configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            Optional[Dict[str, Any]]: The cached response if found, None otherwise

        """
        ...

    @abstractmethod
    async def cache_response(
        self,
        prompt: str,
        model_name: str,
        response: Dict[str, Any],
        config: Optional[ModelCachingConfig] = None,
        trace_id: Optional[str] = None,
    ) -> bool:
        """
        Cache a response for a prompt and model.

        Args:
        ----
            prompt: The prompt to cache
            model_name: The model name
            response: The response to cache
            config: Optional caching configuration
            trace_id: Optional trace ID for monitoring

        Returns:
        -------
            bool: True if caching was successful, False otherwise

        """
        ...
