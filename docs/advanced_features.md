# Advanced Features

Saplings provides a rich set of advanced features for working with modern LLMs, including function calling, vision models, structured output, audio and video processing, and parallel function execution. This document explains how to use these features.

## Function Calling

Function calling allows models to invoke functions defined by the developer. This is useful for tasks like retrieving information, performing calculations, or taking actions based on user input.

### Defining Functions

Functions are defined as JSON objects with a name, description, and parameters:

```python
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
```

### Using Function Calling

To use function calling, pass the function definitions to the `generate` method:

```python
from saplings import LLM

model = LLM.from_uri("openai://gpt-4")

response = await model.generate(
    prompt=[
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ],
    functions=[get_weather_function],
    function_call="auto"  # Let the model decide when to call the function
)

if response.function_call:
    # The model decided to call a function
    function_name = response.function_call["name"]
    function_args = json.loads(response.function_call["arguments"])

    # Call the actual function
    result = call_function(function_name, function_args)

    # Add the function result to the conversation
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": function_name,
                "arguments": response.function_call["arguments"]
            }
        },
        {
            "role": "function",
            "name": function_name,
            "content": json.dumps(result)
        }
    ]

    # Get the final response
    final_response = await model.generate(prompt=messages)
    print(final_response.text)
else:
    # The model responded directly
    print(response.text)
```

### Controlling Function Calling

You can control when functions are called using the `function_call` parameter:

- `"auto"`: Let the model decide when to call a function
- `"none"`: Don't call any functions
- `{"name": "function_name"}`: Always call the specified function

```python
# Always call the get_weather function
response = await model.generate(
    prompt=messages,
    functions=[get_weather_function],
    function_call={"name": "get_weather"}
)
```

### Streaming with Function Calling

Function calling also works with streaming:

```python
async for chunk in model.generate_streaming(
    prompt=messages,
    functions=[get_weather_function],
    function_call="auto"
):
    if isinstance(chunk, dict) and "function_call" in chunk:
        # Process function call chunk
        print(f"Function call: {chunk['function_call']}")
    else:
        # Process text chunk
        print(chunk, end="", flush=True)
```

## Vision Models

Vision models can process both text and images. Saplings supports vision models through the same interface as text-only models.

### Creating Messages with Images

To include images in a message, use a list of content objects:

```python
from saplings import LLM

model = LLM.from_uri("openai://gpt-4-vision-preview")

# Create a message with text and an image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }
]

response = await model.generate(prompt=messages)
print(response.text)
```

### Using Base64-Encoded Images

You can also use base64-encoded images:

```python
import base64

# Load an image file
with open("image.jpg", "rb") as f:
    image_data = f.read()

# Encode the image as base64
image_base64 = base64.b64encode(image_data).decode()

# Create a message with text and an image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    }
]

response = await model.generate(prompt=messages)
print(response.text)
```

### Using the Message Class

Saplings also provides a `Message` class for creating messages with images:

```python
from saplings.core.message import Message, MessageContent, ContentType

# Create a message with text and an image
message = Message.with_image(
    text="What's in this image?",
    image_url="https://example.com/image.jpg"
)

# Or with image data
message = Message.with_image_data(
    text="What's in this image?",
    image_data=image_data
)

# Convert to a dictionary for the model
messages = [message.to_dict()]

response = await model.generate(prompt=messages)
print(response.text)
```

## Structured Output (JSON Mode)

Structured output allows you to get responses in a specific format, such as JSON. This is useful for parsing and processing responses programmatically.

### Using JSON Mode

To use JSON mode, set the `json_mode` parameter to `True`:

```python
from saplings import LLM
import json

model = LLM.from_uri("openai://gpt-4")

response = await model.generate(
    prompt=[
        {"role": "user", "content": "Give me information about the top 3 programming languages in 2023."}
    ],
    json_mode=True
)

# Parse the JSON response
data = json.loads(response.text)
print(data)
```

### Combining JSON Mode with Function Calling

You can combine JSON mode with function calling:

```python
response = await model.generate(
    prompt=messages,
    functions=[get_weather_function],
    function_call="auto",
    json_mode=True
)

if response.function_call:
    # Process function call
    function_args = json.loads(response.function_call["arguments"])
    # ...
else:
    # Parse JSON response
    data = json.loads(response.text)
    # ...
```

## Chat Interface

Saplings provides a chat interface for working with conversation-based models. This interface is designed to be more intuitive than the raw `generate` method.

### Using the Chat Interface

```python
from saplings import LLM

model = LLM.from_uri("openai://gpt-4")

response = await model.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.text)
```

### Streaming with the Chat Interface

```python
async for chunk in model.chat_streaming(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
):
    print(chunk, end="", flush=True)
```

## Provider-Specific Features

Different providers may support different advanced features. Here's a summary of which features are supported by each provider:

| Provider | Function Calling | Vision | JSON Mode |
|----------|-----------------|--------|-----------|
| OpenAI   | ✅              | ✅     | ✅        |
| Anthropic | ✅             | ✅     | ✅        |
| vLLM     | ✅ (limited)    | ❌     | ✅ (limited) |
| HuggingFace | ❌           | ❌     | ❌        |

### OpenAI

OpenAI supports all advanced features:

```python
model = LLM.from_uri("openai://gpt-4")
```

### Anthropic

Anthropic supports all advanced features:

```python
model = LLM.from_uri("anthropic://claude-3-opus-20240229")
```

### vLLM

vLLM has limited support for function calling and JSON mode:

```python
model = LLM.from_uri("vllm://meta-llama/Llama-3.1-8B-Instruct")
```

Function calling with vLLM works by parsing the model's output to extract function calls. This is less reliable than native function calling but can still be useful.

JSON mode with vLLM works by using grammar-based sampling if available.

### HuggingFace

HuggingFace currently does not support advanced features:

```python
model = LLM.from_uri("huggingface://meta-llama/Llama-3-8b-instruct")
```

## Function Registry

Saplings provides a function registry for managing functions that can be called by models. This makes it easy to organize and reuse functions across your application.

### Registering Functions

You can register functions using the `register_function` decorator:

```python
from saplings.core.function_registry import register_function

@register_function(description="Get the weather for a location", group="weather")
def get_weather(location: str, unit: str = "celsius") -> dict:
    """
    Get the weather for a location.

    Args:
        location: The location to get weather for
        unit: The unit to use (celsius or fahrenheit)

    Returns:
        dict: Weather information
    """
    # Implementation...
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "sunny"
    }
```

The function registry automatically extracts parameter information from type hints and docstrings.

### Using Registered Functions

You can get function definitions from the registry and use them with LLMs:

```python
from saplings.core.function_registry import function_registry

# Get function definitions for a group
weather_functions = function_registry.get_group_definitions("weather")

# Use with an LLM
response = await model.generate(
    prompt=messages,
    functions=weather_functions,
    function_call="auto"
)

# Call a registered function
if response.function_call:
    function_name = response.function_call["name"]
    function_args = json.loads(response.function_call["arguments"])

    result = function_registry.call_function(function_name, function_args)
```

## Parallel Function Calling

Saplings provides utilities for calling functions in parallel, which is useful for executing multiple functions concurrently.

### Calling Functions in Parallel

```python
from saplings.core.parallel_function import call_functions_parallel

# Define function calls
function_calls = [
    {"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}},
    {"name": "get_city_info", "arguments": {"city": "Tokyo"}},
    {"name": "convert_currency", "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}},
]

# Call functions in parallel
results = await call_functions_parallel(function_calls)

# Process results
for name, result in results:
    print(f"Result from {name}: {result}")
```

### Setting Timeouts

You can set a timeout for parallel function calls:

```python
# Call functions with a timeout of 5 seconds
results = await call_functions_parallel(function_calls, timeout=5.0)
```

### Using with Async Functions

Parallel function calling works with both synchronous and asynchronous functions:

```python
@register_function(description="Convert currency")
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    # Async implementation...
    await asyncio.sleep(0.2)
    return {"result": amount * 0.85}
```

## Audio and Video Processing

Saplings supports audio and video inputs in addition to text and images.

### Creating Messages with Audio

```python
from saplings.core.message import Message

# Create a message with text and an audio file
message = Message.with_audio(
    text="What's in this audio?",
    audio_url="https://example.com/audio.mp3"
)

# Or with audio data
with open("audio.mp3", "rb") as f:
    audio_data = f.read()

message = Message.with_audio_data(
    text="What's in this audio?",
    audio_data=audio_data
)
```

### Creating Messages with Video

```python
# Create a message with text and a video file
message = Message.with_video(
    text="What's in this video?",
    video_url="https://example.com/video.mp4"
)

# Or with video data
with open("video.mp4", "rb") as f:
    video_data = f.read()

message = Message.with_video_data(
    text="What's in this video?",
    video_data=video_data
)
```

### Using with LLMs

You can use audio and video inputs with LLMs that support them:

```python
# Convert to a dictionary for the model
messages = [message.to_dict()]

response = await model.generate(prompt=messages)
print(response.text)
```

## Streaming Functions

Saplings provides utilities for streaming results from functions, which is useful for long-running operations or generating data incrementally.

### Creating Streaming Functions

You can create streaming functions using Python generators or async generators:

```python
from saplings.core.function_registry import register_function

@register_function(description="Generate a sequence of numbers")
async def generate_sequence(start: int, end: int, delay: float = 0.1):
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
```

### Calling Streaming Functions

You can call streaming functions and process the results as they are generated:

```python
from saplings.core.streaming_function import call_function_streaming

async def process_sequence():
    async for number in call_function_streaming(
        "generate_sequence",
        {"start": 1, "end": 10, "delay": 0.5}
    ):
        print(f"Received: {number}")
```

### Calling Multiple Streaming Functions

You can call multiple streaming functions in parallel:

```python
from saplings.core.streaming_function import call_functions_streaming

async def process_multiple_streams():
    function_calls = [
        {"name": "generate_sequence", "arguments": {"start": 1, "end": 5}},
        {"name": "generate_letters", "arguments": {"count": 5}}
    ]

    async for result in call_functions_streaming(function_calls):
        for name, value in result.items():
            print(f"From {name}: {value}")
```

## Function Validation

Saplings provides utilities for validating function calls, which helps ensure that functions are called with valid arguments.

### Validating Function Calls

You can validate function calls before executing them:

```python
from saplings.core.function_validation import validate_function_call

# Validate arguments
try:
    validated_args = validate_function_call(
        "get_weather",
        {"location": "San Francisco", "unit": "celsius"}
    )

    # Call the function with validated arguments
    result = function_registry.call_function("get_weather", validated_args)
except ValidationError as e:
    print(f"Validation error: {e}")
```

The validation system automatically converts types when possible (e.g., string to int) and checks that required parameters are provided.

## Function Logging

Saplings provides utilities for logging function calls, which helps with debugging, monitoring, and auditing.

### Logging Function Calls

You can log function calls with their arguments and results:

```python
from saplings.core.function_logging import log_function_call

# Log a function call
log_entry = log_function_call(
    name="get_weather",
    arguments={"location": "San Francisco"},
    result={"temperature": 22, "unit": "celsius"},
    metadata={"source": "user_request"}
)
```

### Timing Function Calls

You can time function calls using a context manager:

```python
from saplings.core.function_logging import time_function_call

# Time a function call
with time_function_call(
    name="expensive_calculation",
    arguments={"a": 10, "b": 20}
):
    result = expensive_calculation(10, 20)
```

## Function Caching

Saplings provides utilities for caching function calls, which helps improve performance by avoiding redundant computations.

### Caching Function Results

You can cache function results using the `@cached` decorator:

```python
from saplings.core.function_caching import cached

@register_function(description="Perform an expensive calculation")
@cached(ttl=60)  # Cache for 60 seconds
def expensive_calculation(a: int, b: int) -> int:
    """Perform an expensive calculation."""
    print(f"Calculating {a} * {b}...")
    time.sleep(1)  # Simulate an expensive operation
    return a * b
```

### Controlling Cache Behavior

You can control cache behavior with various parameters:

```python
@cached(
    ttl=3600,           # Cache for 1 hour
    max_size=1000,      # Store up to 1000 items
    namespace="math",   # Use a custom namespace
    key_generator=lambda a, b: f"mul_{a}_{b}"  # Custom key generator
)
def multiply(a: int, b: int) -> int:
    return a * b
```

### Clearing the Cache

You can clear the cache when needed:

```python
from saplings.core.function_caching import clear_cache, clear_all_caches

# Clear a specific cache
clear_cache("math")

# Clear all caches
clear_all_caches()
```

## Function Authorization

Saplings provides utilities for authorizing function calls, which helps control access to sensitive functions.

### Setting Authorization Levels

You can set authorization levels for functions and groups:

```python
from saplings.core.function_authorization import (
    AuthorizationLevel,
    set_function_level,
    set_group_level
)

# Set levels for individual functions
set_function_level("get_user_info", AuthorizationLevel.USER)
set_function_level("get_sensitive_data", AuthorizationLevel.ADMIN)

# Set levels for function groups
set_group_level("public_api", AuthorizationLevel.PUBLIC)
set_group_level("admin_api", AuthorizationLevel.ADMIN)
```

### Using the Authorization Decorator

You can use the `@requires_level` decorator to set authorization levels:

```python
from saplings.core.function_authorization import requires_level, AuthorizationLevel

@register_function(description="Get sensitive data")
@requires_level(AuthorizationLevel.ADMIN)
def get_sensitive_data(user_id: str) -> dict:
    """Get sensitive data for a user."""
    return {"user_id": user_id, "api_key": "secret"}
```

### Setting the Current Authorization Level

You can set the current authorization level based on the user's role:

```python
from saplings.core.function_authorization import set_current_level, AuthorizationLevel

# Set the current level based on user role
if user.is_admin:
    set_current_level(AuthorizationLevel.ADMIN)
elif user.is_authenticated:
    set_current_level(AuthorizationLevel.USER)
else:
    set_current_level(AuthorizationLevel.PUBLIC)
```

### Getting Authorized Functions

You can get a list of functions that the current level is authorized to call:

```python
from saplings.core.function_authorization import get_authorized_functions, get_authorized_groups

# Get authorized functions and groups
authorized_functions = get_authorized_functions()
authorized_groups = get_authorized_groups()
```

## Conclusion

Saplings provides a rich set of advanced features for working with modern LLMs. These features allow you to build more powerful and flexible applications that can interact with the world, process multimodal inputs, execute functions in parallel, validate inputs, log operations, cache results, and control access.

For more information, see the [API reference](./api_reference.md) and the [examples](../examples) directory.
