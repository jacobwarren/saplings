"""
Streaming function module for Saplings.

This module provides utilities for streaming results from functions.
"""

import asyncio
import inspect
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from saplings.core.function_registry import function_registry

logger = logging.getLogger(__name__)


class StreamingFunctionCaller:
    """Utility for calling functions that stream results."""
    
    def __init__(self):
        """Initialize the streaming function caller."""
        pass
    
    async def call_function_streaming(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> AsyncGenerator[Any, None]:
        """
        Call a function and stream the results.
        
        Args:
            name: Name of the function to call
            arguments: Arguments to pass to the function
            
        Yields:
            Any: Results as they are generated
            
        Raises:
            ValueError: If the function is not registered or doesn't support streaming
        """
        # Get the function
        func_info = function_registry.get_function(name)
        if not func_info:
            raise ValueError(f"Function not registered: {name}")
        
        func = func_info["function"]
        
        # Check if the function is a generator or async generator
        if inspect.isasyncgenfunction(func):
            # Function is an async generator
            async for result in func(**arguments):
                yield result
        elif inspect.isgeneratorfunction(func):
            # Function is a generator
            gen = func(**arguments)
            try:
                while True:
                    try:
                        result = await asyncio.to_thread(next, gen)
                        yield result
                    except StopIteration:
                        break
            finally:
                gen.close()
        else:
            # Function doesn't support streaming
            raise ValueError(f"Function {name} doesn't support streaming")
    
    async def call_functions_streaming(
        self,
        function_calls: List[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Call multiple functions and stream the results.
        
        Args:
            function_calls: List of function calls, each with "name" and "arguments" keys
            
        Yields:
            Dict[str, Any]: Results as they are generated, with function name as key
            
        Raises:
            ValueError: If any function is not registered or doesn't support streaming
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
            
            # Create a task for each function call
            task = asyncio.create_task(self._stream_function(name, arguments))
            tasks.append((name, task))
        
        # Process results as they come in
        pending = {task for _, task in tasks}
        function_names = {task: name for name, task in tasks}
        
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                name = function_names[task]
                try:
                    result = task.result()
                    yield {name: result}
                except Exception as e:
                    logger.error(f"Error calling function {name}: {e}")
                    yield {name: {"error": str(e)}}
    
    async def _stream_function(self, name: str, arguments: Dict[str, Any]) -> List[Any]:
        """
        Stream a function and collect the results.
        
        Args:
            name: Name of the function to call
            arguments: Arguments to pass to the function
            
        Returns:
            List[Any]: List of results
            
        Raises:
            ValueError: If the function is not registered or doesn't support streaming
        """
        results = []
        async for result in self.call_function_streaming(name, arguments):
            results.append(result)
        return results


# Create a singleton instance
streaming_function_caller = StreamingFunctionCaller()


async def call_function_streaming(
    name: str,
    arguments: Dict[str, Any]
) -> AsyncGenerator[Any, None]:
    """
    Call a function and stream the results.
    
    This is a convenience function that uses the StreamingFunctionCaller.
    
    Args:
        name: Name of the function to call
        arguments: Arguments to pass to the function
        
    Yields:
        Any: Results as they are generated
        
    Raises:
        ValueError: If the function is not registered or doesn't support streaming
    """
    async for result in streaming_function_caller.call_function_streaming(name, arguments):
        yield result


async def call_functions_streaming(
    function_calls: List[Dict[str, Any]]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Call multiple functions and stream the results.
    
    This is a convenience function that uses the StreamingFunctionCaller.
    
    Args:
        function_calls: List of function calls, each with "name" and "arguments" keys
        
    Yields:
        Dict[str, Any]: Results as they are generated, with function name as key
        
    Raises:
        ValueError: If any function is not registered or doesn't support streaming
    """
    async for result in streaming_function_caller.call_functions_streaming(function_calls):
        yield result
