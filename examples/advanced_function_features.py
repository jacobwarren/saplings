"""
Example of using advanced function features in Saplings.

This example demonstrates how to use streaming functions, validation,
logging, caching, and authorization.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

from saplings.core.function_authorization import (
    AuthorizationLevel,
    requires_level,
    set_current_level,
    set_group_level,
)
from saplings.core.function_caching import cached
from saplings.core.function_logging import time_function_call
from saplings.core.function_registry import register_function
from saplings.core.function_validation import validate_function_call
from saplings.core.streaming_function import call_function_streaming

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Register a streaming function
@register_function(description="Generate a sequence of numbers", group="streaming")
async def generate_sequence(start: int, end: int, delay: float = 0.1) -> AsyncGenerator[int, None]:
    """
    Generate a sequence of numbers with a delay between each.
    
    Args:
        start: Starting number
        end: Ending number
        delay: Delay between numbers in seconds
        
    Yields:
        int: Numbers in the sequence
    """
    for i in range(start, end + 1):
        await asyncio.sleep(delay)
        yield i


# Register a cached function
@register_function(description="Perform an expensive calculation", group="math")
@cached(ttl=60)  # Cache for 60 seconds
def expensive_calculation(a: int, b: int) -> int:
    """
    Perform an expensive calculation.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        int: Result of the calculation
    """
    print(f"Performing expensive calculation: {a} * {b}")
    time.sleep(1)  # Simulate an expensive operation
    return a * b


# Register a function with authorization
@register_function(description="Get sensitive data", group="data")
@requires_level(AuthorizationLevel.ADMIN)
def get_sensitive_data(user_id: str) -> Dict[str, str]:
    """
    Get sensitive data for a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        Dict[str, str]: Sensitive data
    """
    return {
        "user_id": user_id,
        "email": f"{user_id}@example.com",
        "api_key": "secret-api-key",
    }


# Register a normal function
@register_function(description="Get user info", group="data")
def get_user_info(user_id: str) -> Dict[str, str]:
    """
    Get information about a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        Dict[str, str]: User information
    """
    return {
        "user_id": user_id,
        "name": f"User {user_id}",
        "role": "user",
    }


async def run_streaming_example():
    """Run an example of streaming functions."""
    print("=== Streaming Functions Example ===")
    
    print("Streaming a sequence of numbers:")
    async for number in call_function_streaming(
        "generate_sequence",
        {"start": 1, "end": 5, "delay": 0.2}
    ):
        print(f"  Received: {number}")


async def run_validation_example():
    """Run an example of function validation."""
    print("\n=== Function Validation Example ===")
    
    # Validate valid arguments
    try:
        validated_args = validate_function_call(
            "expensive_calculation",
            {"a": "10", "b": "20"}  # String values will be converted to integers
        )
        print(f"Validated arguments: {validated_args}")
    except Exception as e:
        print(f"Validation error: {e}")
    
    # Validate invalid arguments
    try:
        validate_function_call(
            "expensive_calculation",
            {"a": "not_a_number", "b": 20}
        )
        print("Validation succeeded (unexpected)")
    except Exception as e:
        print(f"Validation error (expected): {e}")


async def run_logging_example():
    """Run an example of function logging."""
    print("\n=== Function Logging Example ===")
    
    # Log a function call using a context manager
    with time_function_call(
        "expensive_calculation",
        {"a": 5, "b": 10},
        metadata={"source": "example"}
    ):
        result = expensive_calculation(5, 10)
        print(f"Result: {result}")


async def run_caching_example():
    """Run an example of function caching."""
    print("\n=== Function Caching Example ===")
    
    # First call (cache miss)
    start_time = time.time()
    result1 = expensive_calculation(30, 40)
    end_time = time.time()
    print(f"First call: {result1} (took {end_time - start_time:.2f} seconds)")
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = expensive_calculation(30, 40)
    end_time = time.time()
    print(f"Second call: {result2} (took {end_time - start_time:.2f} seconds)")
    
    # Different arguments (cache miss)
    start_time = time.time()
    result3 = expensive_calculation(50, 60)
    end_time = time.time()
    print(f"Different args: {result3} (took {end_time - start_time:.2f} seconds)")


async def run_authorization_example():
    """Run an example of function authorization."""
    print("\n=== Function Authorization Example ===")
    
    # Set authorization level for a group
    set_group_level("data", AuthorizationLevel.USER)
    
    # Set current level to USER
    set_current_level(AuthorizationLevel.USER)
    print("Current level: USER")
    
    # Try to call a function that requires USER level
    try:
        result = get_user_info("123")
        print(f"User info: {result}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    
    # Try to call a function that requires ADMIN level
    try:
        result = get_sensitive_data("123")
        print(f"Sensitive data: {result}")
    except PermissionError as e:
        print(f"Permission error (expected): {e}")
    
    # Set current level to ADMIN
    set_current_level(AuthorizationLevel.ADMIN)
    print("\nCurrent level: ADMIN")
    
    # Try again with ADMIN level
    try:
        result = get_sensitive_data("123")
        print(f"Sensitive data: {result}")
    except PermissionError as e:
        print(f"Permission error: {e}")


async def main():
    """Run the examples."""
    await run_streaming_example()
    await run_validation_example()
    await run_logging_example()
    await run_caching_example()
    await run_authorization_example()


if __name__ == "__main__":
    asyncio.run(main())
