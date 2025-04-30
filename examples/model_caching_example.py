"""
Example of using model response caching in Saplings.

This example demonstrates how to use caching to improve performance
when making repeated requests to language models.
"""

import asyncio
import time
from typing import Dict, List, Optional

from saplings.core.model_adapter import LLM
from saplings.core.model_caching import clear_model_cache, get_model_cache


async def run_basic_caching_example():
    """Run a basic example of model caching."""
    print("=== Basic Model Caching Example ===")
    
    # Create a model
    model = LLM.from_uri("openai://gpt-3.5-turbo")
    
    # Generate a response without caching
    print("Generating first response (no cache)...")
    start_time = time.time()
    response1 = await model.generate(
        prompt="Explain the concept of caching in computer science in one paragraph."
    )
    end_time = time.time()
    print(f"Response: {response1.text}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Generate the same response with caching
    print("\nGenerating second response (with cache)...")
    start_time = time.time()
    response2 = await model.generate_with_cache(
        prompt="Explain the concept of caching in computer science in one paragraph."
    )
    end_time = time.time()
    print(f"Response: {response2.text}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Generate a different response
    print("\nGenerating third response (different prompt, no cache hit)...")
    start_time = time.time()
    response3 = await model.generate_with_cache(
        prompt="Explain the concept of virtualization in computer science in one paragraph."
    )
    end_time = time.time()
    print(f"Response: {response3.text}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")


async def run_chat_caching_example():
    """Run an example of caching with chat."""
    print("\n=== Chat Caching Example ===")
    
    # Create a model
    model = LLM.from_uri("openai://gpt-3.5-turbo")
    
    # Create a conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the benefits of caching in web applications?"}
    ]
    
    # Generate a response without caching
    print("Generating first response (no cache)...")
    start_time = time.time()
    response1 = await model.chat(
        messages=messages
    )
    end_time = time.time()
    print(f"Response: {response1.text}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Generate the same response with caching
    print("\nGenerating second response (with cache)...")
    start_time = time.time()
    response2 = await model.chat(
        messages=messages,
        use_cache=True
    )
    end_time = time.time()
    print(f"Response: {response2.text}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")


async def run_cache_control_example():
    """Run an example of controlling the cache."""
    print("\n=== Cache Control Example ===")
    
    # Create a model
    model = LLM.from_uri("openai://gpt-3.5-turbo")
    
    # Generate a response with a custom namespace
    print("Generating response with custom namespace...")
    response1 = await model.generate_with_cache(
        prompt="What is the capital of France?",
        cache_namespace="geography"
    )
    print(f"Response: {response1.text}")
    
    # Check if the response is in the cache
    cache = get_model_cache("geography")
    print(f"Response in cache: {cache.get('some_key') is not None}")
    
    # Clear the cache
    print("Clearing the cache...")
    clear_model_cache("geography")
    
    # Generate the same response again
    print("Generating response again after clearing cache...")
    start_time = time.time()
    response2 = await model.generate_with_cache(
        prompt="What is the capital of France?",
        cache_namespace="geography"
    )
    end_time = time.time()
    print(f"Response: {response2.text}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")


async def run_cache_ttl_example():
    """Run an example of cache TTL."""
    print("\n=== Cache TTL Example ===")
    
    # Create a model
    model = LLM.from_uri("openai://gpt-3.5-turbo")
    
    # Generate a response with a short TTL
    print("Generating response with short TTL (1 second)...")
    response1 = await model.generate_with_cache(
        prompt="What is the capital of Germany?",
        cache_ttl=1  # 1 second TTL
    )
    print(f"Response: {response1.text}")
    
    # Wait for the cache to expire
    print("Waiting for cache to expire...")
    await asyncio.sleep(2)
    
    # Generate the same response again
    print("Generating response again after TTL expired...")
    start_time = time.time()
    response2 = await model.generate_with_cache(
        prompt="What is the capital of Germany?",
        cache_ttl=1
    )
    end_time = time.time()
    print(f"Response: {response2.text}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")


async def run_function_calling_caching_example():
    """Run an example of caching with function calling."""
    print("\n=== Function Calling Caching Example ===")
    
    # Create a model
    model = LLM.from_uri("openai://gpt-4")
    
    # Define a function
    get_weather_function = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature to use",
            }
        },
        "required": ["location"]
    }
    
    # Generate a response with function calling
    print("Generating first response with function calling (no cache)...")
    start_time = time.time()
    response1 = await model.generate(
        prompt="What's the weather like in San Francisco?",
        functions=[get_weather_function],
        function_call="auto"
    )
    end_time = time.time()
    print(f"Function call: {response1.function_call}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Generate the same response with caching
    print("\nGenerating second response with function calling (with cache)...")
    start_time = time.time()
    response2 = await model.generate_with_cache(
        prompt="What's the weather like in San Francisco?",
        functions=[get_weather_function],
        function_call="auto"
    )
    end_time = time.time()
    print(f"Function call: {response2.function_call}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")


async def main():
    """Run the examples."""
    await run_basic_caching_example()
    await run_chat_caching_example()
    await run_cache_control_example()
    await run_cache_ttl_example()
    await run_function_calling_caching_example()


if __name__ == "__main__":
    asyncio.run(main())
