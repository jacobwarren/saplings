from __future__ import annotations

"""
Tokenizer factory for creating tokenizers on-demand.

This module provides a factory for creating tokenizers, supporting lazy initialization
and dependency injection to break circular dependencies.
"""

import logging
from typing import Optional

from saplings._internal.tokenizer import TokenizerInterface
from saplings._internal.tokenizers.simple_tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)

# Shadow model availability flag
SHADOW_MODEL_AVAILABLE = False


# Define a placeholder for ShadowModelTokenizer
class ShadowModelTokenizer:
    """Placeholder for ShadowModelTokenizer."""


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
        # Shadow model tokenizer is not available, fall back to simple tokenizer
        logger.warning("Shadow model tokenizer not available, falling back to simple tokenizer")
        return cls.create_simple_tokenizer(**kwargs)
