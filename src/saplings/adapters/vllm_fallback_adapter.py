from __future__ import annotations

"""
VLLM Fallback Adapter for environments without Triton support.

This module provides a fallback adapter for vLLM that can be used in environments
where Triton is not available, such as Apple Silicon Macs.
"""


import logging
from typing import TYPE_CHECKING, Any

from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelRole

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class VLLMFallbackAdapter(LLM):
    """
    VLLM Fallback Adapter for environments without Triton support.

    This adapter provides a fallback for vLLM in environments where Triton is not available.
    It delegates to another model provider (OpenAI, Anthropic, or HuggingFace) based on availability.
    """

    def __init__(self, provider: str, model_name: str, **kwargs) -> None:
        """
        Initialize the VLLM fallback adapter.

        Args:
        ----
            provider: The model provider
            model_name: The model name
            **kwargs: Additional parameters for the model

        """
        super().__init__(provider, model_name, **kwargs)

        # Store provider and model name
        self.provider = provider
        self.model_name = model_name

        # Default parameters
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.temperature = kwargs.get("temperature", 0.7)

        # Try to determine the best fallback provider
        self.fallback_provider = kwargs.get("fallback_provider")
        self.fallback_model = kwargs.get("fallback_model")

        # If no fallback provider is specified, try to determine the best one
        if not self.fallback_provider:
            self._determine_fallback_provider()

        # Create the fallback model
        self._create_fallback_model()

        logger.info(
            f"Initialized VLLM fallback adapter with provider: {self.fallback_provider}, "
            f"model: {self.fallback_model}"
        )

    def _determine_fallback_provider(self) -> None:
        """Determine the best fallback provider based on availability."""
        # Try OpenAI first
        try:
            import openai

            self.fallback_provider = "openai"
            self.fallback_model = "gpt-3.5-turbo"
            return
        except ImportError:
            pass

        # Try Anthropic next
        try:
            import anthropic

            self.fallback_provider = "anthropic"
            self.fallback_model = "claude-instant-1"
            return
        except ImportError:
            pass

        # Try HuggingFace as a last resort
        try:
            import transformers

            self.fallback_provider = "huggingface"
            self.fallback_model = "distilgpt2"  # Small model that should work everywhere
            return
        except ImportError:
            pass

        # If all else fails, use a simple fallback
        self.fallback_provider = "simple"
        self.fallback_model = "fallback"
        logger.warning(
            "No suitable fallback provider found. Using a simple fallback that will return "
            "placeholder responses. Please install openai, anthropic, or transformers."
        )

    def _create_fallback_model(self) -> None:
        """Create the fallback model."""
        # Import here to avoid circular imports
        from saplings.core.model_adapter import LLM

        if self.fallback_provider == "simple":
            # Create a simple fallback that just returns placeholder responses
            self.fallback_model_instance = SimpleFallbackModel()
            return

        # Create the fallback model using the LLM.create method
        try:
            if not self.fallback_provider or not self.fallback_model:
                raise RuntimeError(
                    "Fallback provider and model must be set before creating fallback model instance."
                )
            self.fallback_model_instance = LLM.create(
                provider=self.fallback_provider,
                model_name=self.fallback_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            logger.info(f"Created fallback model: {self.fallback_provider}/{self.fallback_model}")
        except Exception as e:
            logger.exception(f"Error creating fallback model: {e}")
            # Fall back to the simple fallback
            self.fallback_provider = "simple"
            self.fallback_model = "fallback"
            self.fallback_model_instance = SimpleFallbackModel()

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
        # Delegate to the fallback model
        return await self.fallback_model_instance.generate(
            prompt=prompt,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            use_cache=use_cache,
            cache_namespace=cache_namespace,
            cache_ttl=cache_ttl,
            **kwargs,
        )

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
            chunk_size: Number of tokens per chunk
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            **kwargs: Additional arguments for generation

        Yields:
        ------
            Union[str, Dict[str, Any]]: Text chunks or function call chunks as they are generated

        """
        # Delegate to the fallback model
        # The generate_streaming method returns an async generator directly
        # We don't need to await it, just iterate through it
        async for chunk in self.fallback_model_instance.generate_streaming(
            prompt=prompt,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            chunk_size=chunk_size,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            **kwargs,
        ):
            yield chunk

    def get_metadata(self) -> Any:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Metadata about the model

        """
        # Delegate to the fallback model
        return self.fallback_model_instance.get_metadata()

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
        # Delegate to the fallback model
        return self.fallback_model_instance.estimate_tokens(text)

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
        # Delegate to the fallback model
        return self.fallback_model_instance.estimate_cost(prompt_tokens, completion_tokens)


class SimpleFallbackModel(LLM):
    """
    Simple fallback model that returns placeholder responses.

    This is used as a last resort when no other fallback is available.
    """

    def __init__(self) -> None:
        """Initialize the simple fallback model."""
        super().__init__("simple", "fallback")
        self.provider = "simple"
        self.model_name = "fallback"
        self.max_tokens = 2048
        self.temperature = 0.7

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
        # Create a simple response based on the prompt
        if isinstance(prompt, str):
            response_text = f"This is a fallback response to: {prompt[:100]}..."
        else:
            # Extract the last user message
            user_messages = [m for m in prompt if m.get("role") == "user"]
            if user_messages:
                last_user_message = user_messages[-1].get("content", "")
                response_text = f"This is a fallback response to: {last_user_message[:100]}..."
            else:
                response_text = "This is a fallback response."

        # Handle function calls if requested
        function_call_result = None
        tool_calls_result = None

        if functions and (function_call == "auto" or isinstance(function_call, dict)):
            # Create a simple function call response
            if isinstance(function_call, dict) and "name" in function_call:
                function_name = function_call["name"]
                function_call_result = {
                    "name": function_name,
                    "arguments": "{}",
                }
            elif functions:
                # Pick the first function
                function_name = functions[0]["name"]
                function_call_result = {
                    "name": function_name,
                    "arguments": "{}",
                }

        # Create the response
        return LLMResponse(
            text=response_text if not function_call_result and not tool_calls_result else None,
            provider=self.provider,
            model_name=self.model_name,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            metadata={
                "model": self.model_name,
                "provider": self.provider,
            },
            function_call=function_call_result,
            tool_calls=tool_calls_result,
        )

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
            chunk_size: Number of tokens per chunk
            functions: List of function definitions that the model may call
            function_call: Controls how the model calls functions
                           Can be "auto", "none", or {"name": "function_name"}
            json_mode: Whether to force the model to output valid JSON
            **kwargs: Additional arguments for generation

        Yields:
        ------
            Union[str, Dict[str, Any]]: Text chunks or function call chunks as they are generated

        """
        # Get a non-streaming response and yield it as a single chunk
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

    def get_metadata(self) -> ModelMetadata:
        """
        Get metadata about the model.

        Returns
        -------
            ModelMetadata: Metadata about the model

        """
        return ModelMetadata(
            name="fallback",
            provider="simple",
            version="1.0",
            description="Simple fallback model for environments without proper LLM support",
            context_window=4096,
            max_tokens_per_request=2048,
            cost_per_1k_tokens_input=0.0,
            cost_per_1k_tokens_output=0.0,
            roles=[ModelRole.GENERAL, ModelRole.JUDGE],
        )

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
        # Simple estimation: 1 token per 4 characters
        return len(text) // 4

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
        # No cost for the fallback model
        return 0.0
