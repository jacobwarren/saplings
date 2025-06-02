from __future__ import annotations

"""
Public API for utility functions.

This module provides the public API for utility functions used throughout Saplings.
"""

from typing import List, Optional

from saplings.api.stability import stable
from saplings.core.utils import (
    count_tokens as _count_tokens,
)
from saplings.core.utils import (
    get_tokens_remaining as _get_tokens_remaining,
)
from saplings.core.utils import (
    is_apple_silicon as _is_apple_silicon,
)
from saplings.core.utils import (
    is_triton_available as _is_triton_available,
)
from saplings.core.utils import (
    split_text_by_tokens as _split_text_by_tokens,
)
from saplings.core.utils import (
    truncate_text_tokens as _truncate_text_tokens,
)
from saplings.utils._internal import async_run_sync as _async_run_sync
from saplings.utils._internal import get_model_sync as _get_model_sync
from saplings.utils._internal import sync_run_sync as _run_sync

__all__ = [
    "run_sync",
    "async_run_sync",
    "get_model_sync",
    "count_tokens",
    "get_tokens_remaining",
    "split_text_by_tokens",
    "truncate_text_tokens",
    "is_apple_silicon",
    "is_triton_available",
]


@stable
def run_sync(coro_func, *args, **kwargs):
    """
    Run an async coroutine function in a synchronous context.

    This function should only be used at the highest level of your application
    when you need to call async code from a synchronous context. In most cases,
    you should structure your code to use async/await throughout.

    Args:
    ----
        coro_func: Async coroutine function to run
        *args: Arguments to pass to the coroutine
        **kwargs: Keyword arguments to pass to the coroutine

    Returns:
    -------
        Any: Result of the coroutine

    Raises:
    ------
        Exception: Any exception raised by the coroutine

    """
    return _run_sync(coro_func, *args, **kwargs)


@stable
def async_run_sync(coro, *args, timeout=None, **kwargs):
    """
    Run an async coroutine in a synchronous context with timeout support.

    This function should only be used at the highest level of your application
    when you need to call async code from a synchronous context. In most cases,
    you should structure your code to use async/await throughout.

    Args:
    ----
        coro: Async coroutine function to run
        *args: Arguments to pass to the coroutine
        timeout: Optional timeout in seconds
        **kwargs: Keyword arguments to pass to the coroutine

    Returns:
    -------
        Any: Result of the coroutine

    Raises:
    ------
        OperationTimeoutError: If the operation times out
        OperationCancelledError: If the operation is cancelled
        Exception: Any other exception raised by the coroutine

    """
    return _async_run_sync(coro, *args, timeout=timeout, **kwargs)


@stable
def get_model_sync(model_service):
    """
    Get a model synchronously from a model service.

    This is a convenience function for getting a model from a model service
    in a synchronous context. It should only be used at the highest level
    of your application when you need to call async code from a synchronous
    context.

    Args:
    ----
        model_service: The model service to get the model from

    Returns:
    -------
        Any: The model

    Raises:
    ------
        Exception: Any exception raised by the get_model method

    """
    return _get_model_sync(model_service)


@stable
def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Count the number of tokens in a text string.

    Args:
    ----
        text: The text to count tokens for
        model_name: The name of the model to use for tokenization

    Returns:
    -------
        The number of tokens in the text

    """
    return _count_tokens(text, model_name)


@stable
def get_tokens_remaining(text: str, max_tokens: int, model_name: Optional[str] = None) -> int:
    """
    Get the number of tokens remaining after a text string.

    Args:
    ----
        text: The text to count tokens for
        max_tokens: The maximum number of tokens allowed
        model_name: The name of the model to use for tokenization

    Returns:
    -------
        The number of tokens remaining

    """
    return _get_tokens_remaining(text, max_tokens, model_name)


@stable
def split_text_by_tokens(
    text: str, chunk_size: int, overlap: int = 0, model_name: Optional[str] = None
) -> List[str]:
    """
    Split a text string into chunks of a specified token size.

    Args:
    ----
        text: The text to split
        chunk_size: The maximum number of tokens per chunk
        overlap: The number of tokens to overlap between chunks
        model_name: The name of the model to use for tokenization

    Returns:
    -------
        A list of text chunks

    """
    # The internal function expects overlap_tokens parameter
    return _split_text_by_tokens(text, chunk_size, model_name, overlap)


@stable
def truncate_text_tokens(text: str, max_tokens: int, model_name: Optional[str] = None) -> str:
    """
    Truncate a text string to a specified number of tokens.

    Args:
    ----
        text: The text to truncate
        max_tokens: The maximum number of tokens to keep
        model_name: The name of the model to use for tokenization

    Returns:
    -------
        The truncated text

    """
    return _truncate_text_tokens(text, max_tokens, model_name)


@stable
def is_apple_silicon() -> bool:
    """
    Check if the current system is running on Apple Silicon.

    Returns
    -------
        True if running on Apple Silicon, False otherwise

    """
    return _is_apple_silicon()


@stable
def is_triton_available() -> bool:
    """
    Check if Triton is available for GPU acceleration.

    Returns
    -------
        True if Triton is available, False otherwise

    """
    return _is_triton_available()
