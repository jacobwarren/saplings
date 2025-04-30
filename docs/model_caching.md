# Model Response Caching

Saplings provides a caching system for model responses, which can significantly improve performance and reduce costs when making repeated requests to language models.

## Basic Usage

To use model response caching, you can use the `generate_with_cache` method or set `use_cache=True` in the `generate` method:

```python
from saplings.core.model_adapter import LLM
import asyncio

async def main():
    model = LLM.from_uri("openai://gpt-4")

    # Using generate_with_cache
    response = await model.generate_with_cache(
        prompt="What is the capital of France?"
    )

    # Or using generate with use_cache=True
    response = await model.generate(
        prompt="What is the capital of France?",
        use_cache=True
    )

# Run the async function
asyncio.run(main())
```

The first time you make a request, the model will generate a response and cache it. Subsequent identical requests will return the cached response without making a new API call, which is much faster and doesn't incur additional costs.

## Caching with Chat

You can also use caching with the chat interface:

```python
from saplings.core.model_adapter import LLM
import asyncio

async def main():
    model = LLM.from_uri("openai://gpt-4")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    # Using chat with use_cache=True
    response = await model.chat(
        messages=messages,
        use_cache=True
    )

    print(response.text)

# Run the async function
asyncio.run(main())
```

## Cache Control

### Cache Namespaces

You can use different cache namespaces to organize your cached responses:

```python
from saplings.core.model_adapter import LLM
import asyncio

async def main():
    model = LLM.from_uri("openai://gpt-4")

    # Using a custom namespace
    response = await model.generate_with_cache(
        prompt="What is the capital of France?",
        cache_namespace="geography"
    )

    print(response.text)

# Run the async function
asyncio.run(main())
```

### Cache TTL (Time to Live)

You can set a time-to-live (TTL) for cached responses:

```python
from saplings.core.model_adapter import LLM
import asyncio

async def main():
    model = LLM.from_uri("openai://gpt-4")

    # Cache for 1 hour (3600 seconds)
    response1 = await model.generate_with_cache(
        prompt="What is the capital of France?",
        cache_ttl=3600
    )

    # Cache for 1 day
    response2 = await model.generate_with_cache(
        prompt="What is the capital of Germany?",
        cache_ttl=86400
    )

    # Cache indefinitely (no expiration)
    response3 = await model.generate_with_cache(
        prompt="What is the capital of Italy?",
        cache_ttl=None
    )

    print(response1.text)
    print(response2.text)
    print(response3.text)

# Run the async function
asyncio.run(main())
```

### Clearing the Cache

You can clear the cache when needed:

```python
from saplings.core.model_caching import clear_model_cache, clear_all_model_caches

# Clear a specific namespace
clear_model_cache("geography")

# Clear all caches
clear_all_model_caches()
```

## Advanced Usage

### Cache Keys

Cache keys are generated based on the model URI, prompt, and all generation parameters. This ensures that responses are only reused when all relevant parameters are the same.

For example, these requests will use different cache keys:

```python
from saplings.core.model_adapter import LLM
import asyncio

async def main():
    model = LLM.from_uri("openai://gpt-4")

    # Different prompts
    response1 = await model.generate_with_cache(prompt="What is the capital of France?")
    response2 = await model.generate_with_cache(prompt="What is the capital of Germany?")

    # Different temperatures
    response3 = await model.generate_with_cache(prompt="What is the capital of France?", temperature=0.7)
    response4 = await model.generate_with_cache(prompt="What is the capital of France?", temperature=0.5)

    # Different max_tokens
    response5 = await model.generate_with_cache(prompt="What is the capital of France?", max_tokens=100)
    response6 = await model.generate_with_cache(prompt="What is the capital of France?", max_tokens=200)

    # These will all use different cache keys
    print(f"Response 1: {response1.text[:20]}...")
    print(f"Response 2: {response2.text[:20]}...")
    print(f"Response 3: {response3.text[:20]}...")
    print(f"Response 4: {response4.text[:20]}...")
    print(f"Response 5: {response5.text[:20]}...")
    print(f"Response 6: {response6.text[:20]}...")

# Run the async function
asyncio.run(main())
```

### Caching with Function Calling

Caching works with function calling as well:

```python
from saplings.core.model_adapter import LLM
import asyncio

async def main():
    model = LLM.from_uri("openai://gpt-4")

    get_weather_function = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"]
        }
    }

    # Generate a response with function calling
    response = await model.generate_with_cache(
        prompt="What's the weather like in San Francisco?",
        functions=[get_weather_function],
        function_call="auto"
    )

    # The function call will be cached
    print(response.function_call)

# Run the async function
asyncio.run(main())
```

### Direct Cache Access

You can access the cache directly if needed:

```python
from saplings.core.model_caching import get_model_cache
import asyncio

async def main():
    # Get a cache
    cache = get_model_cache("my_namespace")

    # Check if a key exists
    if await cache.get("some_key") is not None:
        print("Key exists in cache")

    # Create a response to cache
    from saplings.core.model_adapter import LLMResponse
    some_response = LLMResponse(
        text="This is a cached response.",
        model_uri="openai://gpt-4",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        metadata={"model": "gpt-4", "provider": "openai"}
    )

    # Set a value manually (not recommended)
    await cache.set("some_key", some_response)

    # Delete a key
    await cache.delete("some_key")

    # Clear the cache
    await cache.clear()

# Run the async function
asyncio.run(main())
```

## Performance Considerations

### Cache Size

By default, each cache namespace has a maximum size of 1000 items. When the cache is full, the least recently used items are removed to make space for new items.

You can change the maximum size when getting a cache:

```python
from saplings.core.model_caching import get_model_cache
import asyncio

async def main():
    # Get a cache with a custom size
    cache = get_model_cache("my_namespace", max_size=10000)

    # Use the cache
    # ...

# Run the async function
asyncio.run(main())
```

### Memory Usage

Cached responses are stored in memory, so be mindful of memory usage when caching large responses or using a large cache size.

### Cache Hits vs. Misses

You can monitor cache performance by checking the logs. Cache hits and misses are logged at the DEBUG level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now cache hits and misses will be logged
```

## Best Practices

### When to Use Caching

Caching is most effective when:

- You make the same requests repeatedly
- The responses don't need to be real-time or frequently updated
- You want to reduce API costs
- You want to improve response times

### When Not to Use Caching

Caching may not be appropriate when:

- Responses need to be unique or non-deterministic
- Responses need to be real-time or frequently updated
- You're using streaming responses

### Cache Invalidation

Consider when and how to invalidate your cache:

- Use appropriate TTLs based on how frequently the data changes
- Clear specific namespaces when the underlying data changes
- Use different namespaces for different types of data

## Conclusion

Model response caching is a powerful feature that can significantly improve performance and reduce costs when using language models. By caching responses, you can avoid making redundant API calls and provide faster responses to users.

For more information, see the [API reference](./api_reference.md) and the [examples](../examples) directory.
