from __future__ import annotations

"""
saplings.services.model_caching_service.
====================================

Encapsulates model caching functionality:
- Response caching
- Cache configuration
- Cost estimation
"""


import logging
from typing import Any, Dict, List, Optional, Union

from saplings.api.core.interfaces import IModelCachingService, IModelInitializationService
from saplings.core._internal.exceptions import ModelError
from saplings.core._internal.model_interface import LLM, LLMResponse
from saplings.core.resilience import with_timeout

logger = logging.getLogger(__name__)


class ModelCachingService(IModelCachingService):
    """
    Service that handles model caching.

    This service is responsible for caching model responses and estimating costs.
    """

    def __init__(
        self,
        model_initialization_service: IModelInitializationService,
        cache_enabled: bool = True,
        cache_namespace: str = "model",
        cache_ttl: Optional[int] = 3600,
        cache_provider: str = "memory",
        cache_strategy: Optional[str] = None,
    ) -> None:
        """
        Initialize the model caching service.

        Args:
        ----
            model_initialization_service: Service for model initialization
            cache_enabled: Whether to enable caching
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds (None for no expiration)
            cache_provider: Cache provider to use
            cache_strategy: Cache eviction strategy ("lru", "lfu", "fifo")

        """
        self._model_initialization_service = model_initialization_service
        self.cache_enabled = cache_enabled
        self.cache_namespace = cache_namespace
        self.cache_ttl = cache_ttl
        self.cache_provider = cache_provider
        self.cache_strategy = cache_strategy

        logger.info("ModelCachingService initialized")

    async def get_model(self, timeout: Optional[float] = None) -> LLM:
        """
        Get the LLM instance from the initialization service.

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
        return await self._model_initialization_service.get_model(timeout=timeout)

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
        try:
            # Get model metadata with timeout
            metadata = await self._model_initialization_service.get_model_metadata(timeout=timeout)

            # Define a function that performs the cost calculation
            async def _calculate_cost():
                cost_per_1k_tokens_input = metadata.get("cost_per_1k_tokens_input", 0)
                cost_per_1k_tokens_output = metadata.get("cost_per_1k_tokens_output", 0)

                # Calculate cost
                input_cost = (prompt_tokens / 1000.0) * cost_per_1k_tokens_input
                output_cost = (completion_tokens / 1000.0) * cost_per_1k_tokens_output

                return input_cost + output_cost

            # Execute with timeout
            return await with_timeout(
                _calculate_cost(), timeout=timeout, operation_name="estimate_cost"
            )
        except Exception as e:
            # Wrap the original exception to provide context while preserving the trace
            if not isinstance(e, ModelError):
                msg = f"Failed to estimate cost: {e!s}"
                raise ModelError(msg, cause=e)
            raise

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
        try:
            # Get the model asynchronously with timeout
            model = await self.get_model(timeout=timeout)

            # Define the generation function
            async def _generate():
                try:
                    if self.cache_enabled:
                        response = await model.generate_with_cache(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            cache_namespace=self.cache_namespace,
                            cache_ttl=self.cache_ttl,
                            cache_provider=self.cache_provider,
                            cache_strategy=self.cache_strategy,
                            **kwargs,
                        )
                    else:
                        response = await model.generate(
                            prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
                        )
                    return response.text or ""
                except Exception as e:
                    # Wrap the original exception to provide context while preserving the trace
                    msg = f"Failed to generate text: {e!s}"
                    raise ModelError(msg, cause=e)

            # Execute with timeout
            return await with_timeout(_generate(), timeout=timeout, operation_name="generate_text")
        except Exception as e:
            # Wrap the original exception to provide context while preserving the trace
            if not isinstance(e, ModelError):
                msg = f"Failed to generate text: {e!s}"
                raise ModelError(msg, cause=e)
            raise

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
        try:
            # Get the model asynchronously with timeout
            model = await self.get_model(timeout=timeout)

            # Define the generation function
            async def _generate_response():
                try:
                    if self.cache_enabled:
                        return await model.generate_with_cache(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            cache_namespace=self.cache_namespace,
                            cache_ttl=self.cache_ttl,
                            cache_provider=self.cache_provider,
                            cache_strategy=self.cache_strategy,
                            **kwargs,
                        )
                    return await model.generate(
                        prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
                    )
                except Exception as e:
                    # Wrap the original exception to provide context while preserving the trace
                    msg = f"Failed to generate response: {e!s}"
                    raise ModelError(msg, cause=e)

            # Execute with timeout
            return await with_timeout(
                _generate_response(), timeout=timeout, operation_name="generate_response"
            )
        except Exception as e:
            # Wrap the original exception to provide context while preserving the trace
            if not isinstance(e, ModelError):
                msg = f"Failed to generate response: {e!s}"
                raise ModelError(msg, cause=e)
            raise
