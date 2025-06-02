from __future__ import annotations

"""
Utility functions for synchronous operations.

This module provides utility functions for running async code in synchronous contexts.
These functions should only be used at the highest level of your application when
you need to call async code from a synchronous context. In most cases, you should
structure your code to use async/await throughout.
"""

import asyncio
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_sync(coro_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
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
    try:
        # Create a new event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the coroutine and return the result
        coro = coro_func(*args, **kwargs)
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.exception(f"Error running async function synchronously: {e}")
        raise


def get_model_sync(model_service: Any) -> Any:
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
    return run_sync(model_service.get_model)
