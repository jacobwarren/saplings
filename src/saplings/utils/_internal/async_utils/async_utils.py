from __future__ import annotations

"""
Utility functions for async operations.

This module provides utility functions for working with async code,
particularly for cases where async code needs to be called from sync contexts.
"""

import asyncio
import logging
from typing import Any, Callable, TypeVar

from saplings.core._internal.exceptions import OperationCancelledError, OperationTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_sync(
    coro: Callable[..., Any], *args: Any, timeout: float | None = None, **kwargs: Any
) -> Any:
    """
    Run an async coroutine in a synchronous context.

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
    # Get or create an event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If there's no event loop in this thread, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Create a simple async function that applies the timeout
    async def run_with_timeout():
        try:
            if timeout is not None:
                return await asyncio.wait_for(coro(*args, **kwargs), timeout=timeout)
            else:
                return await coro(*args, **kwargs)
        except asyncio.TimeoutError:
            raise OperationTimeoutError(f"Operation timed out after {timeout} seconds")
        except asyncio.CancelledError:
            raise OperationCancelledError("Operation was cancelled")

    # Run the coroutine with proper error handling
    if loop.is_running():
        # If we're already in an event loop, we can't use run_until_complete
        # This is a common issue when using this function in a context that's already async
        logger.warning(
            "Attempting to run an async function synchronously from within an async context. "
            "This is not recommended and may cause issues. Consider restructuring your code "
            "to use async/await throughout."
        )

        # Create a new event loop for a new thread
        new_loop = asyncio.new_event_loop()

        # Define a function to run in the thread
        def thread_target():
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(run_with_timeout())

        # Run the function in a thread with a timeout
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(thread_target)
            try:
                # Wait for the result with a timeout
                return future.result(timeout=timeout if timeout is not None else 30.0)
            except concurrent.futures.TimeoutError:
                logger.error(f"Thread execution timed out after {timeout} seconds")
                # We can't reliably stop the thread, but we can try to cancel any tasks
                try:
                    for task in asyncio.all_tasks(new_loop):
                        task.cancel()
                except Exception:
                    pass
                raise OperationTimeoutError(f"Operation timed out after {timeout} seconds")
            except Exception as e:
                logger.error(f"Error in thread execution: {type(e).__name__}: {e}")
                if isinstance(e, TimeoutError):
                    raise OperationTimeoutError(str(e))
                raise
    else:
        # If we're not in an event loop, run the coroutine in this thread
        try:
            return loop.run_until_complete(run_with_timeout())
        except Exception as e:
            logger.error(f"Error in run_until_complete: {type(e).__name__}: {e}")
            if isinstance(e, TimeoutError):
                raise OperationTimeoutError(str(e))
            raise
