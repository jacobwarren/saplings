"""
This example demonstrates how to use Saplings' model caching capabilities
to improve performance and reduce costs when making repeated LLM calls.
"""

from __future__ import annotations

import asyncio
import time

from saplings import Agent, AgentConfig
from saplings.core.caching import CacheConfig, ModelCache
from saplings.core.model_adapter import LLM


async def main():
    # Create a model with caching enabled
    print("Setting up model cache...")
    cache_config = CacheConfig(
        enable=True,
        ttl=3600,  # Cache entries expire after 1 hour
        max_size=1000,  # Maximum entries in the cache
        storage_path="./model_cache",  # Persist cache to disk
    )

    # Create model with caching
    print("Creating model with caching...")
    model = LLM.create("openai", "gpt-4o")
    model_cache = ModelCache(config=cache_config)
    model.set_cache(model_cache)

    # Create an agent with the cached model
    print("Creating agent...")
    agent = Agent(
        config=AgentConfig(
            provider="openai",
            model_name="gpt-4o",
            enable_model_caching=True,
            # Pass cache config parameters directly
            cache_ttl=cache_config.ttl,
            cache_max_size=cache_config.max_size,
            cache_storage_path=cache_config.storage_path,
        )
    )

    # Alternatively, set the model directly
    # agent.model = model

    # Create model cache directory if it doesn't exist
    if not os.path.exists("./model_cache"):
        os.makedirs("./model_cache")
        print("Created model cache directory")

    # Demonstration queries (some similar, some different)
    print("Preparing test queries...")
    queries = [
        "What is the capital of France?",
        "What is the population of Paris?",
        "What is the capital of France?",  # Exact duplicate (should hit cache)
        "What's the capital city of France?",  # Semantic duplicate (should hit cache)
        "Tell me about the Eiffel Tower",
        "What is the square root of 144?",
        "Calculate the square root of 144",  # Semantic duplicate
    ]

    # Run queries and measure performance
    print("\nRunning queries with caching enabled:")

    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")

        # Time the execution
        start_time = time.time()
        result = await agent.run(query)
        end_time = time.time()

        # Get cache info
        cache_hit = model_cache.was_last_request_cached()

        # Print results
        print(f"Result: {result[:100]}..." if len(result) > 100 else f"Result: {result}")
        print(f"Time: {(end_time - start_time):.4f} seconds")
        print(f"Cache: {'HIT' if cache_hit else 'MISS'}")

    # Print cache statistics
    stats = model_cache.get_statistics()
    print("\nCache Statistics:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Estimated tokens saved: {stats['tokens_saved']:,}")
    print(f"Estimated cost saved: ${stats['cost_saved']:.4f}")

    # Clear cache and run without caching for comparison
    print("\nClearing cache and disabling caching for comparison...")
    model_cache.clear()
    model_cache.disable()

    # Run the same queries without caching
    print("\nRunning queries with caching disabled:")
    total_time_without_cache = 0

    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")

        # Time the execution
        start_time = time.time()
        result = await agent.run(query)
        end_time = time.time()
        execution_time = end_time - start_time
        total_time_without_cache += execution_time

        # Print results
        print(f"Result: {result[:100]}..." if len(result) > 100 else f"Result: {result}")
        print(f"Time: {execution_time:.4f} seconds")

    # Compare overall performance
    print("\nPerformance Comparison:")
    total_time_with_cache = stats["total_time"]
    print(f"Total time with cache: {total_time_with_cache:.4f} seconds")
    print(f"Total time without cache: {total_time_without_cache:.4f} seconds")
    print(f"Time saved: {(total_time_without_cache - total_time_with_cache):.4f} seconds")
    print(
        f"Performance improvement: {((total_time_without_cache - total_time_with_cache) / total_time_without_cache):.2%}"
    )


if __name__ == "__main__":
    import os  # Added import for directory creation

    asyncio.run(main())
