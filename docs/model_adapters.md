# Model Adapters

The Model Adapters system in Saplings provides a unified interface to different LLM providers, enabling seamless integration with various models while maintaining a consistent API.

## Overview

The Model Adapters system consists of several key components:

- **LLM**: Abstract base class for all model adapters
- **ModelService**: Manages model initialization and operations
- **ModelAdapter**: Implementations for different providers (OpenAI, Anthropic, vLLM, etc.)
- **ModelRegistry**: Registers and manages model instances

This system enables applications to work with different LLM providers without changing the core code, providing flexibility and future-proofing.

## Core Concepts

### LLM Interface

The `LLM` class is the abstract base class that all model adapters must implement:

- **Initialization**: Common parameters like provider, model name, and additional options
- **Generation**: Methods for generating text from prompts
- **Chat**: Methods for generating responses to conversations
- **Streaming**: Methods for streaming responses
- **Function Calling**: Support for function/tool calling capabilities

### Model Configuration

Models are configured using a provider, model name, and optional parameters:

- **Provider**: The model provider (e.g., "openai", "anthropic", "vllm")
- **Model Name**: The name of the model (e.g., "gpt-4o", "claude-3-opus-20240229")
- **Parameters**: Additional parameters like temperature, max_tokens, etc.

### Model Registry

The Model Registry manages model instances:

- **Registration**: Models are registered with a unique key
- **Resolution**: Models are resolved by key
- **Caching**: Models are cached to avoid duplicate instances
- **Lifecycle Management**: Models are properly disposed when no longer needed

### Provider Adapters

Provider adapters implement the LLM interface for specific providers:

- **OpenAIAdapter**: For OpenAI models (GPT-4, GPT-3.5, etc.)
- **AnthropicAdapter**: For Anthropic models (Claude 3, etc.)
- **VLLMAdapter**: For models served through vLLM
- **HuggingFaceAdapter**: For models from Hugging Face
- **TransformersAdapter**: For direct use of Transformers models

## API Reference

### LLM

```python
class LLM(ABC):
    @abstractmethod
    def __init__(self, provider: str, model_name: str, **kwargs):
        """Initialize the LLM adapter."""
        pass

    @classmethod
    def create(cls, provider: str, model_name: str, **kwargs) -> "LLM":
        """Create an LLM instance with a provider and model name."""
        pass


    @abstractmethod
    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from the model."""
        pass

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response to a conversation."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response from the model."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Get metadata about the model."""
        pass
```

### LLMResponse

```python
class LLMResponse(BaseModel):
    text: Optional[str] = Field(None, description="Generated text")
    provider: str = Field(..., description="Provider of the model")
    model_name: str = Field(..., description="Name of the model")
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics (prompt_tokens, completion_tokens, total_tokens)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )
    function_call: Optional[Dict[str, Any]] = Field(
        None, description="Function call information if the model decided to call a function"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool call information if the model decided to call tools"
    )

    @property
    def content(self) -> Optional[str]:
        """Get the content of the response."""
        return self.text
```

### ModelService

```python
class ModelService:
    def __init__(
        self,
        provider: str,
        model_name: str,
        retry_config: Optional[Dict[str, Any]] = None,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
        cache_enabled: bool = True,
        cache_namespace: str = "model",
        cache_ttl: Optional[int] = 3600,
        cache_provider: str = "memory",
        cache_strategy: Optional[str] = None,
        **model_parameters,
    ):
        """Initialize the model service."""

    def _init_model(self) -> None:
        """Initialize the model."""

    async def get_model(self, timeout: Optional[float] = None) -> LLM:
        """Get the LLM instance."""

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from the model."""

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response to a conversation."""

    async def generate_stream(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response from the model."""
```

## Usage Examples

### Basic Usage

```python
from saplings.core.model_adapter import LLM

# Create a model using the provider and model name
model = LLM.create(provider="openai", model_name="gpt-4o")

# Generate text
import asyncio
response = asyncio.run(model.generate(
    prompt="Explain the concept of graph-based memory.",
    max_tokens=500,
    temperature=0.7,
))

# Print the response
print(response.text)
```

### Using Model Parameters

```python
from saplings.core.model_adapter import LLM

# Create a model with additional parameters
model = LLM.create(
    provider="openai",
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=500,
)

# Generate text
import asyncio
response = asyncio.run(model.generate(
    prompt="Explain the concept of graph-based memory.",
))

# Print the response
print(response.text)
```

### Chat Conversations

```python
from saplings.core.model_adapter import LLM

# Create a model
model = LLM.create(provider="anthropic", model_name="claude-3-opus-20240229")

# Define a conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is graph-based memory?"},
]

# Generate a response
import asyncio
response = asyncio.run(model.chat(
    messages=messages,
    max_tokens=500,
    temperature=0.7,
))

# Print the response
print(response.text)
```

### Function Calling

```python
from saplings.core.model_adapter import LLM

# Create a model
model = LLM.create(provider="openai", model="gpt-4o")

# Define functions
functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use",
                },
            },
            "required": ["location"],
        },
    }
]

# Generate a response with function calling
import asyncio
response = asyncio.run(model.generate(
    prompt="What's the weather like in San Francisco?",
    functions=functions,
    function_call="auto",
))

# Check if a function was called
if response.function_call:
    print(f"Function: {response.function_call['name']}")
    print(f"Arguments: {response.function_call['arguments']}")
else:
    print(response.text)
```

### Streaming Responses

```python
from saplings.core.model_adapter import LLM

# Create a model
model = LLM.create(provider="openai", model_name="gpt-4o")

# Generate a streaming response
import asyncio

async def stream_response():
    async for chunk in model.generate_stream(
        prompt="Write a short story about a robot learning to feel emotions.",
        max_tokens=1000,
        temperature=0.8,
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)

# Run the streaming function
asyncio.run(stream_response())
```

### Using vLLM

```python
from saplings.core.model_adapter import LLM

# Create a vLLM model
model = LLM.create(
    provider="vllm",
    model_name="Qwen/Qwen3-7B-Instruct",
    device="cuda"
)

# Generate text
import asyncio
response = asyncio.run(model.generate(
    prompt="Explain the concept of graph-based memory.",
    max_tokens=500,
    temperature=0.7,
))

# Print the response
print(response.text)
```

### Using Transformers Directly

```python
from saplings.core.model_adapter import LLM

# Create a Transformers model
model = LLM.create(
    provider="transformers",
    model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device="mps"
)

# Generate text
import asyncio
response = asyncio.run(model.generate(
    prompt="Explain the concept of graph-based memory.",
    max_tokens=500,
    temperature=0.7,
))

# Print the response
print(response.text)
```

## Advanced Features

### Model Metadata

```python
from saplings.core.model_adapter import LLM

# Create a model
model = LLM.create(provider="openai", model_name="gpt-4o")

# Get model metadata
metadata = model.metadata

# Print metadata
print(f"Model: {metadata.model_name}")
print(f"Provider: {metadata.provider}")
print(f"Context window: {metadata.context_window}")
print(f"Capabilities: {metadata.capabilities}")
```

### Resilience Features

```python
from saplings.core.model_adapter import LLM
from saplings.services.model_service import ModelService

# Create a model service with resilience features
model_service = ModelService(
    provider="openai",
    model_name="gpt-4o",
    retry_config={
        "max_attempts": 3,
        "initial_backoff": 1.0,
        "max_backoff": 30.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },
    circuit_breaker_config={
        "failure_threshold": 5,
        "recovery_timeout": 30.0,
        "expected_exceptions": ["RateLimitError", "ServiceUnavailableError"],
    },
)

# Get the model
import asyncio
model = asyncio.run(model_service.get_model())

# Generate text with automatic retries and circuit breaking
response = asyncio.run(model_service.generate(
    prompt="Explain the concept of graph-based memory.",
    max_tokens=500,
    temperature=0.7,
))

# Print the response
print(response.text)
```

### Response Caching

```python
from saplings.core.model_adapter import LLM
from saplings.services.model_service import ModelService

# Create a model service with caching
model_service = ModelService(
    provider="openai",
    model_name="gpt-4o",
    cache_enabled=True,
    cache_namespace="my_app",
    cache_ttl=3600,  # 1 hour
    cache_provider="redis",
    cache_strategy="lru",
)

# Generate text with caching
import asyncio
response1 = asyncio.run(model_service.generate(
    prompt="Explain the concept of graph-based memory.",
    max_tokens=500,
    temperature=0.0,  # Use 0 temperature for deterministic responses
    use_cache=True,
))

# This will use the cached response
response2 = asyncio.run(model_service.generate(
    prompt="Explain the concept of graph-based memory.",
    max_tokens=500,
    temperature=0.0,
    use_cache=True,
))

# Print the responses
print(f"Response 1: {response1.text[:50]}...")
print(f"Response 2: {response2.text[:50]}...")
```

## Implementation Details

### OpenAI Adapter

The OpenAI adapter implements the LLM interface for OpenAI models:

1. **Initialization**: Sets up the OpenAI client with API key and other parameters
2. **Message Conversion**: Converts between Saplings message format and OpenAI format
3. **Function Calling**: Handles function calling and tool calling
4. **Streaming**: Implements streaming using OpenAI's streaming API
5. **Error Handling**: Handles OpenAI-specific errors and converts them to Saplings errors

### Anthropic Adapter

The Anthropic adapter implements the LLM interface for Anthropic models:

1. **Initialization**: Sets up the Anthropic client with API key and other parameters
2. **Message Conversion**: Converts between Saplings message format and Anthropic format
3. **Function Calling**: Handles function calling (tool use in Anthropic terminology)
4. **Streaming**: Implements streaming using Anthropic's streaming API
5. **Error Handling**: Handles Anthropic-specific errors and converts them to Saplings errors

### vLLM Adapter

The vLLM adapter implements the LLM interface for models served through vLLM:

1. **Initialization**: Sets up the vLLM client or server
2. **Message Conversion**: Converts between Saplings message format and vLLM format
3. **Function Calling**: Implements function calling using vLLM's function calling capabilities
4. **Streaming**: Implements streaming using vLLM's streaming API
5. **Error Handling**: Handles vLLM-specific errors and converts them to Saplings errors

### HuggingFace Adapter

The HuggingFace adapter implements the LLM interface for models from Hugging Face:

1. **Initialization**: Loads the model and tokenizer from Hugging Face
2. **Message Conversion**: Converts between Saplings message format and HuggingFace format
3. **Function Calling**: Implements function calling using HuggingFace's function calling capabilities
4. **Streaming**: Implements streaming using HuggingFace's streaming API
5. **Error Handling**: Handles HuggingFace-specific errors and converts them to Saplings errors

### Transformers Adapter

The Transformers adapter implements the LLM interface for direct use of Transformers models:

1. **Initialization**: Loads the model and tokenizer directly
2. **Message Conversion**: Converts between Saplings message format and Transformers format
3. **Function Calling**: Implements function calling using Transformers' function calling capabilities
4. **Streaming**: Implements streaming using Transformers' streaming API
5. **Error Handling**: Handles Transformers-specific errors and converts them to Saplings errors

## Extension Points

The Model Adapters system is designed to be extensible:

### Custom Model Adapter

You can create a custom model adapter by implementing the LLM interface:

```python
from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

class CustomModelAdapter(LLM):
    def __init__(self, provider: str, model_name: str, **kwargs):
        """Initialize the custom model adapter."""
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.7)

        # Initialize your custom model here
        self.model = self._load_model(model_name, **kwargs)

    def _load_model(self, model_name: str, **kwargs):
        """Load the model."""
        # Implement your custom model loading logic here
        pass

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from the model."""
        # Implement your custom generation logic here
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Process the prompt
        processed_prompt = self._process_prompt(prompt, functions, function_call, json_mode)

        # Generate text
        generated_text = self._generate_text(processed_prompt, max_tokens, temperature, **kwargs)

        # Create response
        return LLMResponse(
            text=generated_text,
            provider=self.provider,
            model_name=self.model_name,
            usage={
                "prompt_tokens": len(processed_prompt),
                "completion_tokens": len(generated_text),
                "total_tokens": len(processed_prompt) + len(generated_text),
            },
            metadata={
                "custom_field": "custom_value",
            },
        )

    async def generate_stream(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response from the model."""
        # Implement your custom streaming logic here
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Process the prompt
        processed_prompt = self._process_prompt(prompt, functions, function_call, json_mode)

        # Generate text in chunks
        for chunk in self._generate_text_stream(processed_prompt, max_tokens, temperature, **kwargs):
            yield LLMResponse(
                text=chunk,
                provider=self.provider,
                model_name=self.model_name,
                usage={},
                metadata={},
            )

    @property
    def metadata(self) -> ModelMetadata:
        """Get metadata about the model."""
        return ModelMetadata(
            provider=self.provider,
            model_name=self.model_name,
            context_window=4096,
            max_tokens=self.max_tokens,
            capabilities=["text-generation"],
            roles=["assistant"],
        )
```

### Custom Model Service

You can create a custom model service by extending the ModelService class:

```python
from saplings.services.model_service import ModelService
from saplings.core.model_adapter import LLM, LLMResponse
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

class CustomModelService(ModelService):
    def __init__(
        self,
        provider: str,
        model_name: str,
        custom_parameter: str,
        **kwargs,
    ):
        """Initialize the custom model service."""
        super().__init__(provider, model_name, **kwargs)
        self.custom_parameter = custom_parameter

    def _init_model(self) -> None:
        """Initialize the model with custom logic."""
        # Custom initialization logic
        super()._init_model()

        # Additional setup
        if self.model:
            # Customize the model
            pass

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate text with custom pre/post processing."""
        # Custom pre-processing
        processed_prompt = self._preprocess_prompt(prompt)

        # Call the parent method
        response = await super().generate(
            prompt=processed_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            trace_id=trace_id,
            **kwargs,
        )

        # Custom post-processing
        processed_response = self._postprocess_response(response)

        return processed_response

    def _preprocess_prompt(self, prompt: Union[str, List[Dict[str, Any]]]) -> Union[str, List[Dict[str, Any]]]:
        """Custom prompt preprocessing."""
        # Implement your custom preprocessing logic here
        return prompt

    def _postprocess_response(self, response: LLMResponse) -> LLMResponse:
        """Custom response postprocessing."""
        # Implement your custom postprocessing logic here
        return response
```

## Conclusion

The Model Adapters system in Saplings provides a unified interface to different LLM providers, enabling seamless integration with various models while maintaining a consistent API. By implementing the LLM interface for different providers, Saplings ensures that applications can work with any supported model without changing the core code, providing flexibility and future-proofing.
