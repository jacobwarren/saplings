from __future__ import annotations

"""
LLM Builder module for Saplings.

This module provides a builder class for creating LLM instances with
proper configuration. It separates configuration from initialization and
provides a fluent interface for setting configuration parameters.
"""

import logging
from typing import Any, Dict

from saplings.core._internal.exceptions import InitializationError
from saplings.core._internal.model_interface import LLM

logger = logging.getLogger(__name__)


class LLMBuilder:
    """
    Builder for LLM.

    This class provides a fluent interface for building LLM instances with
    proper configuration. It separates configuration from initialization and
    provides a fluent interface for setting configuration parameters.

    Example:
    -------
    ```python
    # Create a builder for LLM
    builder = LLMBuilder()

    # Configure the builder with options
    model = builder.with_provider("openai") \
                  .with_model_name("gpt-4o") \
                  .with_parameters({
                      "temperature": 0.7,
                      "max_tokens": 2048
                  }) \
                  .build()
    ```

    """

    def __init__(self) -> None:
        """Initialize the LLM builder."""
        self._provider = None
        self._model_name = None
        self._params = {}

    def with_provider(self, provider: str) -> LLMBuilder:
        """
        Set the model provider.

        Args:
        ----
            provider: Model provider (e.g., "openai", "anthropic", "vllm")

        Returns:
        -------
            The builder instance for method chaining

        """
        self._provider = provider
        return self

    def with_model_name(self, model_name: str) -> LLMBuilder:
        """
        Set the model name.

        Args:
        ----
            model_name: Model name (e.g., "gpt-4o", "claude-3-opus")

        Returns:
        -------
            The builder instance for method chaining

        """
        self._model_name = model_name
        return self

    def with_parameters(self, params: Dict[str, Any]) -> LLMBuilder:
        """
        Set additional model parameters.

        Args:
        ----
            params: Model parameters

        Returns:
        -------
            The builder instance for method chaining

        """
        self._params.update(params)
        return self

    def with_temperature(self, temperature: float) -> LLMBuilder:
        """
        Set the temperature parameter.

        Args:
        ----
            temperature: Temperature for sampling (0.0 to 1.0)

        Returns:
        -------
            The builder instance for method chaining

        """
        self._params["temperature"] = temperature
        return self

    def with_max_tokens(self, max_tokens: int) -> LLMBuilder:
        """
        Set the maximum number of tokens to generate.

        Args:
        ----
            max_tokens: Maximum number of tokens to generate

        Returns:
        -------
            The builder instance for method chaining

        """
        self._params["max_tokens"] = max_tokens
        return self

    def with_top_p(self, top_p: float) -> LLMBuilder:
        """
        Set the top_p parameter for nucleus sampling.

        Args:
        ----
            top_p: Top-p value for nucleus sampling (0.0 to 1.0)

        Returns:
        -------
            The builder instance for method chaining

        """
        self._params["top_p"] = top_p
        return self

    def with_top_k(self, top_k: int) -> LLMBuilder:
        """
        Set the top_k parameter for top-k sampling.

        Args:
        ----
            top_k: Top-k value for top-k sampling

        Returns:
        -------
            The builder instance for method chaining

        """
        self._params["top_k"] = top_k
        return self

    def with_presence_penalty(self, penalty: float) -> LLMBuilder:
        """
        Set the presence penalty.

        Args:
        ----
            penalty: Presence penalty (-2.0 to 2.0)

        Returns:
        -------
            The builder instance for method chaining

        """
        self._params["presence_penalty"] = penalty
        return self

    def with_frequency_penalty(self, penalty: float) -> LLMBuilder:
        """
        Set the frequency penalty.

        Args:
        ----
            penalty: Frequency penalty (-2.0 to 2.0)

        Returns:
        -------
            The builder instance for method chaining

        """
        self._params["frequency_penalty"] = penalty
        return self

    def with_stop_sequences(self, sequences: list[str]) -> LLMBuilder:
        """
        Set the stop sequences.

        Args:
        ----
            sequences: List of sequences that will stop generation

        Returns:
        -------
            The builder instance for method chaining

        """
        self._params["stop"] = sequences
        return self

    def build(self) -> LLM:
        """
        Build the LLM instance with the configured parameters.

        Returns
        -------
            The initialized LLM instance

        Raises
        ------
            InitializationError: If required parameters are missing

        """
        try:
            if not self._provider:
                raise ValueError("Provider is required")
            if not self._model_name:
                raise ValueError("Model name is required")

            return LLM.create(self._provider, self._model_name, **self._params)
        except Exception as e:
            raise InitializationError(f"Failed to initialize LLM: {e}", cause=e)
