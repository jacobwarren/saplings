from __future__ import annotations

"""
Tokenizer factory for creating tokenizers on-demand.

This module provides a factory for creating tokenizers, supporting lazy initialization
and dependency injection to break circular dependencies.
"""

import logging
from typing import Any, Optional, Type, Union

from saplings.core._internal.tokenizer import TokenizerInterface
from saplings.tokenizers._internal.simple_tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)

# Import shadow model tokenizer if available
try:
    from saplings.tokenizers._internal.shadow.shadow_model_tokenizer import (
        TRANSFORMERS_AVAILABLE,
        ShadowModelTokenizer,
    )

    SHADOW_MODEL_AVAILABLE = True
except ImportError:
    SHADOW_MODEL_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False


class TokenizerFactory:
    """
    Factory for creating tokenizers on-demand.

    This class provides methods for creating tokenizers with various configurations,
    supporting lazy initialization and dependency injection.
    """

    @classmethod
    def create_tokenizer(
        cls,
        tokenizer_type: str = "auto",
        model_name: Optional[str] = None,
        use_fast: bool = True,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        fallback_to_simple: bool = True,
        cpu_only: bool = False,
        alternative_models: Optional[list[str]] = None,
        lazy_init: bool = True,
        **kwargs,
    ) -> TokenizerInterface:
        """
        Create a tokenizer based on the specified type.

        Args:
        ----
            tokenizer_type: Type of tokenizer to create ("simple", "shadow", "auto")
            model_name: Name of the model to use for tokenization (for shadow model tokenizer)
            use_fast: Whether to use the fast tokenizer implementation
            device: Device to use for the model (cpu or cuda)
            cache_dir: Directory to cache the model
            fallback_to_simple: Whether to fall back to SimpleTokenizer if transformers is not available
            cpu_only: Force CPU-only mode even if GPU is available
            alternative_models: List of alternative models to try if the primary model fails
            lazy_init: Whether to initialize the tokenizer lazily
            **kwargs: Additional keyword arguments to pass to the tokenizer

        Returns:
        -------
            TokenizerInterface: The created tokenizer

        """
        if tokenizer_type == "simple":
            return cls.create_simple_tokenizer(**kwargs)
        elif tokenizer_type == "shadow" and SHADOW_MODEL_AVAILABLE:
            return cls.create_shadow_model_tokenizer(
                model_name=model_name,
                use_fast=use_fast,
                device=device,
                cache_dir=cache_dir,
                fallback_to_simple=fallback_to_simple,
                cpu_only=cpu_only,
                alternative_models=alternative_models,
                lazy_init=lazy_init,
                **kwargs,
            )
        elif tokenizer_type == "auto":
            # Try to create a shadow model tokenizer if available, otherwise fall back to simple
            if SHADOW_MODEL_AVAILABLE:
                return cls.create_shadow_model_tokenizer(
                    model_name=model_name,
                    use_fast=use_fast,
                    device=device,
                    cache_dir=cache_dir,
                    fallback_to_simple=True,  # Always fall back to simple for "auto"
                    cpu_only=cpu_only,
                    alternative_models=alternative_models,
                    lazy_init=lazy_init,
                    **kwargs,
                )
            else:
                logger.info("Shadow model tokenizer not available, using simple tokenizer")
                return cls.create_simple_tokenizer(**kwargs)
        else:
            logger.warning(f"Unknown tokenizer type: {tokenizer_type}, using simple tokenizer")
            return cls.create_simple_tokenizer(**kwargs)

    @staticmethod
    def create_simple_tokenizer(vocab_size: int = 50257, **kwargs) -> SimpleTokenizer:
        """
        Create a simple tokenizer.

        Args:
        ----
            vocab_size: Size of the vocabulary
            **kwargs: Additional keyword arguments to pass to the tokenizer

        Returns:
        -------
            SimpleTokenizer: The created tokenizer

        """
        return SimpleTokenizer(vocab_size=vocab_size)

    @classmethod
    def create_shadow_model_tokenizer(
        cls,
        model_name: Optional[str] = None,
        use_fast: bool = True,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        fallback_to_simple: bool = True,
        cpu_only: bool = False,
        alternative_models: Optional[list[str]] = None,
        lazy_init: bool = True,
        **kwargs,
    ) -> TokenizerInterface:
        """
        Create a shadow model tokenizer.

        Args:
        ----
            model_name: Name of the model to use for tokenization
            use_fast: Whether to use the fast tokenizer implementation
            device: Device to use for the model (cpu or cuda)
            cache_dir: Directory to cache the model
            fallback_to_simple: Whether to fall back to SimpleTokenizer if transformers is not available
            cpu_only: Force CPU-only mode even if GPU is available
            alternative_models: List of alternative models to try if the primary model fails
            lazy_init: Whether to initialize the tokenizer lazily
            **kwargs: Additional keyword arguments to pass to the tokenizer

        Returns:
        -------
            TokenizerInterface: The created tokenizer

        """
        if not SHADOW_MODEL_AVAILABLE:
            if fallback_to_simple:
                logger.warning(
                    "Shadow model tokenizer not available, falling back to simple tokenizer"
                )
                return cls.create_simple_tokenizer(**kwargs)
            else:
                msg = "Shadow model tokenizer not available and fallback_to_simple is False"
                raise ImportError(msg)

        # Use default model name if none is provided
        if model_name is None:
            model_name = "Qwen/Qwen3-0.6B"

        # Create the shadow model tokenizer with lazy initialization if requested
        if lazy_init:
            # Create a wrapper that will initialize the tokenizer on first use
            return LazyTokenizer(
                tokenizer_class=ShadowModelTokenizer,
                model_name=model_name,
                use_fast=use_fast,
                device=device,
                cache_dir=cache_dir,
                fallback_to_simple=fallback_to_simple,
                cpu_only=cpu_only,
                alternative_models=alternative_models,
                **kwargs,
            )
        else:
            # Create and initialize the tokenizer immediately
            return ShadowModelTokenizer(
                model_name=model_name,
                use_fast=use_fast,
                device=device,
                cache_dir=cache_dir,
                fallback_to_simple=fallback_to_simple,
                cpu_only=cpu_only,
                alternative_models=alternative_models,
                **kwargs,
            )

    @staticmethod
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
