from __future__ import annotations

"""
Tokenizer factory for creating tokenizers on-demand (part 2).

This module provides additional functionality for the tokenizer factory.
"""

import logging
from typing import Any, Optional, Type, Union

from saplings._internal.tokenizer import TokenizerInterface
from saplings._internal.tokenizers.simple_tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


class LazyTokenizer(TokenizerInterface):
    """
    Lazy-loading wrapper for tokenizers.

    This class wraps a tokenizer class and initializes it only when first used,
    allowing for deferred initialization of expensive resources.
    """

    def __init__(
        self, tokenizer_class: Type[TokenizerInterface], *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize the lazy tokenizer.

        Args:
        ----
            tokenizer_class: Class of the tokenizer to create
            *args: Positional arguments to pass to the tokenizer constructor
            **kwargs: Keyword arguments to pass to the tokenizer constructor

        """
        self._tokenizer_class = tokenizer_class
        self._args = args
        self._kwargs = kwargs
        self._tokenizer: Optional[TokenizerInterface] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize the tokenizer if it hasn't been initialized yet."""
        if not self._initialized:
            logger.debug(f"Lazy-initializing tokenizer of type {self._tokenizer_class.__name__}")
            self._tokenizer = self._tokenizer_class(*self._args, **self._kwargs)
            self._initialized = True

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into tokens.

        Args:
        ----
            text: Text to tokenize

        Returns:
        -------
            List[str]: List of tokens

        """
        self._ensure_initialized()
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]) -> Union[int, list[int]]:
        """
        Convert tokens to token IDs.

        Args:
        ----
            tokens: Token or list of tokens to convert

        Returns:
        -------
            Union[int, List[int]]: Token ID or list of token IDs

        """
        self._ensure_initialized()
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: Union[int, list[int]]) -> Union[str, list[str]]:
        """
        Convert token IDs to tokens.

        Args:
        ----
            ids: Token ID or list of token IDs to convert

        Returns:
        -------
            Union[str, List[str]]: Token or list of tokens

        """
        self._ensure_initialized()
        return self._tokenizer.convert_ids_to_tokens(ids)

    def __call__(self, text: str, return_tensors: Optional[str] = None) -> Any:
        """
        Tokenize text and return a compatible object.

        Args:
        ----
            text: Text to tokenize
            return_tensors: Format of tensors to return

        Returns:
        -------
            Any: Object with input_ids attribute

        """
        self._ensure_initialized()
        return self._tokenizer(text, return_tensors=return_tensors)

    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size.

        Returns
        -------
            int: The vocabulary size

        """
        self._ensure_initialized()
        return self._tokenizer.vocab_size

    @property
    def special_tokens(self) -> dict[str, int]:
        """
        Get the special tokens.

        Returns
        -------
            Dict[str, int]: The special tokens

        """
        self._ensure_initialized()
        return self._tokenizer.special_tokens


def get_tokenizer_for_model(
    model: Any, fallback_to_simple: bool = True
) -> Optional[TokenizerInterface]:
    """
    Get a tokenizer for a specific model.

    This method tries to extract the tokenizer from the model if it has one,
    or creates a new tokenizer that's compatible with the model.

    Args:
    ----
        model: The model to get a tokenizer for
        fallback_to_simple: Whether to fall back to SimpleTokenizer if no tokenizer is found

    Returns:
    -------
        Optional[TokenizerInterface]: The tokenizer for the model, or None if no tokenizer is found
            and fallback_to_simple is False

    """
    # Try to get the tokenizer from the model
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    # If the model doesn't have a tokenizer, try to create one based on the model type
    if fallback_to_simple:
        logger.warning("Model does not have a tokenizer, falling back to simple tokenizer")
        return SimpleTokenizer()
    else:
        logger.warning("Model does not have a tokenizer and fallback_to_simple is False")
        return None
