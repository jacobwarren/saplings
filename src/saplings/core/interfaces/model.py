from __future__ import annotations

"""
Model service interface for Saplings.

This module defines the interface for model services that provide access to
language models and their capabilities. This is a pure interface with no
implementation details or dependencies outside of the core types.
"""


from abc import ABC, abstractmethod
from typing import Any

# Forward references
LLM = Any  # From saplings.core.model_adapter


class IModelService(ABC):
    """Interface for model service operations."""

    @abstractmethod
    async def get_model(self, timeout: float | None = None) -> LLM:
        """
        Get the configured model instance.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            LLM: The model instance

        """

    @abstractmethod
    async def get_model_metadata(self, timeout: float | None = None) -> dict[str, Any]:
        """
        Get metadata about the model.

        Args:
        ----
            timeout: Optional timeout in seconds

        Returns:
        -------
            Dict[str, Any]: Model metadata

        """

    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the model (synchronous version).

        Returns
        -------
            Dict[str, Any]: Model metadata

        """
        # Default implementation for backward compatibility
        import asyncio

        return asyncio.run(self.get_model_metadata())

    @abstractmethod
    async def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, timeout: float | None = None
    ) -> float:
        """
        Estimate the cost of a model call based on token counts.

        Args:
        ----
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            timeout: Optional timeout in seconds

        Returns:
        -------
            float: Estimated cost in USD

        """

    @abstractmethod
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

        """

    @abstractmethod
    async def generate_response(
        self,
        prompt: str | list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_cache: bool = False,
        timeout: float | None = None,
        **kwargs,
    ) -> Any:  # LLMResponse
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

        """

    @abstractmethod
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
    ) -> Any:  # LLMResponse
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

        """
