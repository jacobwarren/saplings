"""
Parallel function calling module for Saplings.

This module provides utilities for calling functions in parallel.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from saplings.core.function_registry import function_registry

logger = logging.getLogger(__name__)


class ParallelFunctionCaller:
    """Utility for calling functions in parallel."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the parallel function caller.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def call_functions(
        self,
        function_calls: List[Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> List[Tuple[str, Any]]:
        """
        Call multiple functions in parallel.
        
        Args:
            function_calls: List of function calls, each with "name" and "arguments" keys
            timeout: Timeout in seconds for all function calls
            
        Returns:
            List[Tuple[str, Any]]: List of (function_name, result) tuples
            
        Raises:
            asyncio.TimeoutError: If the timeout is reached
        """
        # Create tasks for each function call
        tasks = []
        for call in function_calls:
            name = call.get("name")
            arguments = call.get("arguments", {})
            
            if isinstance(arguments, str):
                try:
                    import json
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse arguments for function {name}: {arguments}")
                    arguments = {}
            
            task = self._call_function_async(name, arguments)
            tasks.append(task)
        
        # Run tasks in parallel with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results with function names
        return [(call.get("name"), result) for call, result in zip(function_calls, results)]
    
    async def call_function(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Any:
        """
        Call a single function asynchronously.
        
        Args:
            name: Name of the function to call
            arguments: Arguments to pass to the function
            timeout: Timeout in seconds
            
        Returns:
            Any: The result of the function call
            
        Raises:
            asyncio.TimeoutError: If the timeout is reached
            ValueError: If the function is not registered
        """
        return await self._call_function_async(name, arguments, timeout)
    
    async def _call_function_async(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Any:
        """
        Call a function asynchronously.
        
        Args:
            name: Name of the function to call
            arguments: Arguments to pass to the function
            timeout: Timeout in seconds
            
        Returns:
            Any: The result of the function call
            
        Raises:
            asyncio.TimeoutError: If the timeout is reached
            ValueError: If the function is not registered
        """
        # Get the function
        func_info = function_registry.get_function(name)
        if not func_info:
            raise ValueError(f"Function not registered: {name}")
        
        func = func_info["function"]
        
        # Check if the function is already async
        if asyncio.iscoroutinefunction(func):
            # Call async function directly
            if timeout is not None:
                return await asyncio.wait_for(func(**arguments), timeout)
            return await func(**arguments)
        
        # Run the function in a thread pool
        loop = asyncio.get_event_loop()
        if timeout is not None:
            return await asyncio.wait_for(
                loop.run_in_executor(self.executor, lambda: func(**arguments)),
                timeout
            )
        return await loop.run_in_executor(self.executor, lambda: func(**arguments))


# Create a singleton instance
parallel_function_caller = ParallelFunctionCaller()


async def call_functions_parallel(
    function_calls: List[Dict[str, Any]],
    timeout: Optional[float] = None,
    max_workers: Optional[int] = None
) -> List[Tuple[str, Any]]:
    """
    Call multiple functions in parallel.
    
    This is a convenience function that creates a ParallelFunctionCaller.
    
    Args:
        function_calls: List of function calls, each with "name" and "arguments" keys
        timeout: Timeout in seconds for all function calls
        max_workers: Maximum number of worker threads
        
    Returns:
        List[Tuple[str, Any]]: List of (function_name, result) tuples
        
    Raises:
        asyncio.TimeoutError: If the timeout is reached
    """
    if max_workers is not None:
        caller = ParallelFunctionCaller(max_workers=max_workers)
    else:
        caller = parallel_function_caller
    
    return await caller.call_functions(function_calls, timeout)
