from __future__ import annotations

"""
Public API for tokenizers.

This module provides the public API for tokenizer implementations for various models,
including a simple tokenizer and a shadow model tokenizer for GASA.
It also provides a factory for creating tokenizers on-demand.
"""

from typing import Any, Optional

from saplings._internal.tokenizer import TokenizerInterface
from saplings._internal.tokenizers.factory import TokenizerFactory as _TokenizerFactory

# Import implementation classes
from saplings._internal.tokenizers.simple_tokenizer import SimpleTokenizer as _SimpleTokenizer
from saplings.api.stability import beta, stable
from saplings.api.tokenizers_internal import SHADOW_MODEL_AVAILABLE


@beta
class SimpleTokenizer(_SimpleTokenizer):
    """
    A simple tokenizer for use with GASA when a model-specific tokenizer is not available.

    This tokenizer implements a basic word-level tokenization strategy with special token
    handling. It's not meant to be an accurate representation of any specific model's
    tokenization, but rather a lightweight alternative for GASA mask generation.
    """


@stable
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
        return _TokenizerFactory.create_tokenizer(
            tokenizer_type=tokenizer_type,
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

    @classmethod
    def create_simple_tokenizer(cls, vocab_size: int = 50257, **kwargs) -> TokenizerInterface:
        """
        Create a simple tokenizer.

        Args:
        ----
            vocab_size: Size of the vocabulary
            **kwargs: Additional keyword arguments to pass to the tokenizer

        Returns:
        -------
            TokenizerInterface: The created tokenizer

        """
        return _TokenizerFactory.create_simple_tokenizer(vocab_size=vocab_size, **kwargs)

    @classmethod
    def get_tokenizer_for_model(
        cls, model: Any, fallback_to_simple: bool = True
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
        return _TokenizerFactory.get_tokenizer_for_model(
            model=model, fallback_to_simple=fallback_to_simple
        )


# Import shadow model tokenizer if available
if SHADOW_MODEL_AVAILABLE:
    # Import the shadow model tokenizer
    from saplings.tokenizers._internal.shadow import ShadowModelTokenizer as _ShadowModelTokenizer

    @beta
    class ShadowModelTokenizer(_ShadowModelTokenizer):
        """
        Shadow model tokenizer for GASA with third-party LLM APIs.

        This class loads a small local model for tokenization when working with
        third-party LLM APIs like OpenAI and Anthropic. This allows GASA to work
        effectively even when the underlying LLM is a "black-box" API.
        """

    # Add shadow model tokenizer creation method to TokenizerFactory
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
        return _TokenizerFactory.create_shadow_model_tokenizer(
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

    # Add shadow model tokenizer creation method to TokenizerFactory
    TokenizerFactory.create_shadow_model_tokenizer = create_shadow_model_tokenizer

    __all__ = [
        "SimpleTokenizer",
        "ShadowModelTokenizer",
        "TokenizerFactory",
        "SHADOW_MODEL_AVAILABLE",
    ]
else:
    __all__ = ["SimpleTokenizer", "TokenizerFactory", "SHADOW_MODEL_AVAILABLE"]
