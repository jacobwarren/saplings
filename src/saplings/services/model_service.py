from __future__ import annotations

"""
saplings.services.model_service.
=============================

Encapsulates model initialization, configuration, and management:
- Model creation and URI parsing
- Parameter management
- Registry integration
- Model telemetry and cost estimation
- Caching of model responses
"""


import logging
from typing import Any

from saplings.core.exceptions import (
    ModelError,
    ProviderError,
    ResourceExhaustedError,
)
from saplings.core.interfaces import IModelService
from saplings.core.model_adapter import LLM, LLMResponse
from saplings.core.resilience import CircuitBreaker

logger = logging.getLogger(__name__)


class ModelService(IModelService):
    """
    Service that manages an LLM.

    This service is responsible for model initialization, configuration management,
    and core model operations like metadata access and cost estimation.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        retry_config: dict[str, Any] | None = None,
        circuit_breaker_config: dict[str, Any] | None = None,
        cache_enabled: bool = True,
        cache_namespace: str = "model",
        cache_ttl: int | None = 3600,
        cache_provider: str = "memory",
        cache_strategy: str | None = None,
        **model_parameters,
    ) -> None:
        """
        Initialize the model service.

        Args:
        ----
            provider: Provider name (e.g., "openai", "anthropic")
            model_name: Name of the model
            retry_config: Configuration for retry mechanism
            circuit_breaker_config: Configuration for circuit breaker
            cache_enabled: Whether to enable caching
            cache_namespace: Namespace for the cache
            cache_ttl: Time to live for the cache entry in seconds (None for no expiration)
            cache_provider: Cache provider to use
            cache_strategy: Cache eviction strategy ("lru", "lfu", "fifo")
            **model_parameters: Additional model parameters

        """
        # Set up resilience configuration
        self.retry_config = retry_config or {
            "max_attempts": 3,
            "initial_backoff": 1.0,
            "max_backoff": 30.0,
            "backoff_factor": 2.0,
            "jitter": True,
        }

        # Set up caching configuration
        self.cache_enabled = cache_enabled
        self.cache_namespace = cache_namespace
        self.cache_ttl = cache_ttl
        self.cache_provider = cache_provider
        self.cache_strategy = cache_strategy

        self.circuit_breaker_config = circuit_breaker_config or {
            "failure_threshold": 5,
            "recovery_timeout": 60.0,
        }

        # Create circuit breaker for model calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.circuit_breaker_config["failure_threshold"],
            recovery_timeout=self.circuit_breaker_config["recovery_timeout"],
            expected_exceptions=[
                ProviderError,
                ResourceExhaustedError,
                ConnectionError,
                TimeoutError,
            ],
        )

        self.provider = provider
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.model: LLM | None = None

        # Initialize the model
        self._init_model()

    def _init_model(self):
        """
        Initialize the model.

        Raises
        ------
            ModelError: If model initialization fails

        """
        try:
            # The model registry is handled automatically in LLM.create
            # to ensure that only one instance with the same configuration is created
            self.model = LLM.create(
                provider=self.provider,
                model_name=self.model_name,
                **self.model_parameters,
            )
            logger.info("Model initialized: %s/%s", self.provider, self.model_name)
        except Exception as e:
            # Wrap the original exception to provide context while preserving the trace
            msg = f"Failed to initialize model: {e!s}"
            raise ModelError(msg, cause=e)

    async def get_model(self, timeout: float | None = None) -> LLM:
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
        from saplings.core.resilience import run_in_executor

        # This is a lightweight operation, but we make it async for interface consistency
        def _get_model():
            if self.model is None:
                msg = "Model is not initialized. Call _init_model() first."
                raise ModelError(msg)
            return self.model

        return await run_in_executor(_get_model, timeout=timeout)

    async def get_model_metadata(self, timeout: float | None = None) -> dict[str, Any]:
        """
        Get metadata about the model.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            Dict: Model metadata

        Raises:
        ------
            ModelError: If metadata cannot be retrieved
            OperationTimeoutError: If the operation times out
            OperationCancelledError: If the operation is cancelled

        """
        from saplings.core.resilience import run_in_executor

        try:
            # Get model first (this uses run_in_executor internally now)
            model = await self.get_model(timeout=timeout)

            # Define function to run in thread pool
            def _get_metadata():
                try:
                    metadata = model.get_metadata()
                    # Convert ModelMetadata to dict if needed
                    if metadata is not None:
                        # Import ModelMetadata for type checking
                        from saplings.core.model_adapter import ModelMetadata

                        if isinstance(metadata, ModelMetadata):
                            return metadata.model_dump()
                        if isinstance(metadata, dict):
                            return metadata
                    return metadata or {}
                except Exception as e:
                    # Wrap the original exception to provide context while preserving the trace
                    msg = f"Failed to get model metadata: {e!s}"
                    raise ModelError(msg, cause=e)

            # Run in executor with timeout
            result = await run_in_executor(_get_metadata, timeout=timeout)

            # Ensure we return a Dict[str, Any]
            from saplings.core.model_adapter import ModelMetadata

            if isinstance(result, ModelMetadata):
                return result.model_dump()
            if isinstance(result, dict):
                return result
            # Convert any other type to dict
            return {"value": str(result)}
        except Exception as e:
            # Wrap the original exception to provide context while preserving the trace
            if not isinstance(e, ModelError):
                msg = f"Failed to get model metadata: {e!s}"
                raise ModelError(msg, cause=e)
            raise

    async def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, timeout: float | None = None
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
        from saplings.core.resilience import with_timeout

        # This operation involves an async call to get_model_metadata, which already
        # has timeout handling, and then does a simple calculation
        try:
            # Get model metadata with timeout
            metadata = await self.get_model_metadata(timeout=timeout)

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

    async def generate_text(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_cache: bool = False,
        timeout: float | None = None,
        **kwargs,
    ) -> str:
        """
        Generate text from the model.

        This is a convenience method that handles retries and circuit breaking.

        Args:
        ----
            prompt: The prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            use_cache: Whether to use caching for this request
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
        from saplings.core.resilience import with_timeout

        try:
            # Get the model asynchronously with timeout
            model = await self.get_model(timeout=timeout)

            # Define the generation function
            async def _generate():
                try:
                    if use_cache and self.cache_enabled:
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

    async def generate(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        functions: list[dict[str, Any]] | None = None,
        function_call: str | dict[str, Any] | None = None,
        system_prompt: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from the model.

        This is the main method for generating responses from the model.
        It supports both text generation and function calling.

        Args:
        ----
            prompt: The prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            functions: Optional list of function definitions
            function_call: Optional function call specification
            system_prompt: Optional system prompt
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
        from saplings.core.resilience import with_timeout

        try:
            # Get the model asynchronously with timeout
            model = await self.get_model(timeout=timeout)

            # Define the generation function
            async def _generate():
                try:
                    # Prepare generation parameters
                    gen_kwargs = {
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        **kwargs,
                    }

                    # Add optional parameters if provided
                    if functions is not None:
                        gen_kwargs["functions"] = functions
                    if function_call is not None:
                        gen_kwargs["function_call"] = function_call
                    if system_prompt is not None:
                        gen_kwargs["system_prompt"] = system_prompt

                    # Generate response
                    return await model.generate(**gen_kwargs)
                except Exception as e:
                    # Wrap the original exception to provide context while preserving the trace
                    msg = f"Failed to generate response: {e!s}"
                    raise ModelError(msg, cause=e)

            # Execute with timeout
            return await with_timeout(_generate(), timeout=timeout, operation_name="generate")
        except Exception as e:
            # Wrap the original exception to provide context while preserving the trace
            if not isinstance(e, ModelError):
                msg = f"Failed to generate response: {e!s}"
                raise ModelError(msg, cause=e)
            raise

    async def generate_response(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_cache: bool = False,
        timeout: float | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a full response from the model.

        This is similar to generate_text but returns the complete LLMResponse object
        which includes metadata, token usage, and any function calls.

        Args:
        ----
            prompt: The prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            use_cache: Whether to use caching for this request
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
        from saplings.core.resilience import with_timeout

        try:
            # Get the model asynchronously with timeout
            model = await self.get_model(timeout=timeout)

            # Define the generation function
            async def _generate_response():
                try:
                    if use_cache and self.cache_enabled:
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
