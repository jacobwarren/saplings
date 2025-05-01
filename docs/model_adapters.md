# Model Adapters

Saplings provides a flexible model adapter system that allows you to use different LLM providers for inference. This document explains how to use the various model adapters available in Saplings.

## Overview

The model adapter system in Saplings is designed to provide a unified interface to different LLM providers. This allows you to easily switch between providers without changing your code.

All model adapters implement the `LLM` abstract base class, which defines methods for text generation, streaming, and metadata retrieval.

## Using Model Adapters

### Creating a Model from a URI

The easiest way to create a model is to use the `LLM.from_uri` method, which takes a model URI and returns an instance of the appropriate adapter:

```python
from saplings.core.model_adapter import LLM

# Create a vLLM model
model = LLM.from_uri("vllm://meta-llama/Llama-3.1-8B-Instruct")

# Create an OpenAI model
model = LLM.from_uri("openai://gpt-4")

# Create an Anthropic model
model = LLM.from_uri("anthropic://claude-3-opus-20240229")

# Create a HuggingFace model
model = LLM.from_uri("huggingface://meta-llama/Llama-3-8b-instruct")
```

### Model URI Format

The model URI format is:

```
provider://model_name/version?param1=value1&param2=value2
```

Where:
- `provider` is the model provider (e.g., 'vllm', 'openai', 'anthropic', 'huggingface')
- `model_name` is the name of the model
- `version` is the model version (optional, defaults to 'latest')
- `parameters` are additional parameters for the model (optional)

Examples:
- `vllm://meta-llama/Llama-3.1-8B-Instruct?temperature=0.7&max_tokens=1024`
- `openai://gpt-4/latest?temperature=0.7`
- `anthropic://claude-3-opus-20240229/latest?temperature=0.7`
- `huggingface://meta-llama/Llama-3-8b-instruct/latest?device=cuda`

### Generating Text

Once you have a model, you can use it to generate text:

```python
import asyncio
from saplings.core.model_adapter import LLM

async def generate_text():
    model = LLM.from_uri("vllm://meta-llama/Llama-3.1-8B-Instruct")

    # Generate text
    response = await model.generate(
        prompt="Explain the concept of self-improving AI in simple terms.",
        max_tokens=1024,
        temperature=0.7,
    )

    print(response.text)
    print(f"Token usage: {response.usage}")

asyncio.run(generate_text())
```

### Streaming Generation

You can also generate text with streaming output:

```python
import asyncio
from saplings.core.model_adapter import LLM

async def generate_streaming():
    model = LLM.from_uri("vllm://meta-llama/Llama-3.1-8B-Instruct")

    # Generate text with streaming
    async for chunk in model.generate_streaming(
        prompt="Explain the concept of self-improving AI in simple terms.",
        max_tokens=1024,
        temperature=0.7,
    ):
        print(chunk, end="", flush=True)

asyncio.run(generate_streaming())
```

## Available Adapters

### vLLM Adapter

The vLLM adapter provides high-performance inference using [vLLM](https://github.com/vllm-project/vllm), which is optimized for serving LLMs with GPU acceleration.

```python
from saplings.core.model_adapter import LLM

# Create a vLLM model
model = LLM.from_uri("vllm://meta-llama/Llama-3.1-8B-Instruct")

# With parameters
model = LLM.from_uri("vllm://meta-llama/Llama-3.1-8B-Instruct?temperature=0.7&max_tokens=1024&quantization=awq")

# Or create it directly
from saplings.adapters.vllm_adapter import VLLMAdapter
model = VLLMAdapter("vllm://meta-llama/Llama-3.1-8B-Instruct")
```

#### Installation

To use the vLLM adapter, you need to install vLLM:

```bash
pip install vllm
```

#### Parameters

The vLLM adapter supports the following parameters:

- `temperature`: Temperature for sampling (default: 0.7)
- `max_tokens`: Maximum number of tokens to generate (default: 1024)
- `quantization`: Quantization method to use (e.g., 'awq', 'gptq')
- `trust_remote_code`: Whether to trust remote code (default: True)
- `enable_tool_choice`: Whether to enable function/tool calling (default: True)
- `tool_call_parser`: Parser to use for tool calls (default: None, uses built-in parser)
- `chat_template`: Chat template to use (default: None, auto-detected for Llama 3.1/3.2)

### OpenAI Adapter

The OpenAI adapter provides access to OpenAI's models, such as GPT-4 and GPT-3.5-Turbo.

```python
from saplings.core.model_adapter import LLM

# Create an OpenAI model
model = LLM.from_uri("openai://gpt-4")

# With parameters
model = LLM.from_uri("openai://gpt-4?temperature=0.7&max_tokens=1024&api_key=your-api-key")

# Or create it directly
from saplings.adapters.openai_adapter import OpenAIAdapter
model = OpenAIAdapter("openai://gpt-4")
```

#### Installation

To use the OpenAI adapter, you need to install the OpenAI Python package:

```bash
pip install openai
```

#### Parameters

The OpenAI adapter supports the following parameters:

- `temperature`: Temperature for sampling (default: 0.7)
- `max_tokens`: Maximum number of tokens to generate (default: 1024)
- `api_key`: OpenAI API key (default: uses OPENAI_API_KEY environment variable)
- `api_base`: OpenAI API base URL (default: uses OPENAI_API_BASE environment variable)
- `organization`: OpenAI organization ID (default: uses OPENAI_ORGANIZATION environment variable)

### Anthropic Adapter

The Anthropic adapter provides access to Anthropic's Claude models.

```python
from saplings.core.model_adapter import LLM

# Create an Anthropic model
model = LLM.from_uri("anthropic://claude-3-opus-20240229")

# With parameters
model = LLM.from_uri("anthropic://claude-3-opus-20240229?temperature=0.7&max_tokens=1024&api_key=your-api-key")

# Or create it directly
from saplings.adapters.anthropic_adapter import AnthropicAdapter
model = AnthropicAdapter("anthropic://claude-3-opus-20240229")
```

#### Installation

To use the Anthropic adapter, you need to install the Anthropic Python package:

```bash
pip install anthropic
```

#### Parameters

The Anthropic adapter supports the following parameters:

- `temperature`: Temperature for sampling (default: 0.7)
- `max_tokens`: Maximum number of tokens to generate (default: 1024)
- `api_key`: Anthropic API key (default: uses ANTHROPIC_API_KEY environment variable)

### HuggingFace Adapter

The HuggingFace adapter provides access to models from the HuggingFace Hub.

```python
from saplings.core.model_adapter import LLM

# Create a HuggingFace model
model = LLM.from_uri("huggingface://meta-llama/Llama-3-8b-instruct")

# With parameters
model = LLM.from_uri("huggingface://meta-llama/Llama-3-8b-instruct?temperature=0.7&max_tokens=1024&device=cuda&torch_dtype=float16")

# Or create it directly
from saplings.adapters.huggingface_adapter import HuggingFaceAdapter
model = HuggingFaceAdapter("huggingface://meta-llama/Llama-3-8b-instruct")
```

#### Installation

To use the HuggingFace adapter, you need to install the Transformers and PyTorch packages:

```bash
pip install transformers torch
```

#### Parameters

The HuggingFace adapter supports the following parameters:

- `temperature`: Temperature for sampling (default: 0.7)
- `max_tokens`: Maximum number of tokens to generate (default: 1024)
- `device`: Device to use for inference (default: 'cuda' if available, otherwise 'cpu')
- `torch_dtype`: PyTorch data type to use (default: 'float16' if device is 'cuda', otherwise 'float32')
- `trust_remote_code`: Whether to trust remote code (default: True)

## Creating Custom Adapters

You can create your own model adapters by implementing the `LLM` abstract base class:

```python
from saplings.core.model_adapter import LLM, LLMResponse, ModelMetadata, ModelURI, ModelCapability, ModelRole
from typing import AsyncGenerator, Optional, Union, List, Dict, Any

class MyCustomAdapter(LLM):
    def __init__(self, model_uri: Union[str, ModelURI], **kwargs):
        # Parse the model URI
        if isinstance(model_uri, str):
            self.model_uri = ModelURI.parse(model_uri)
        else:
            self.model_uri = model_uri

        # Initialize your model
        # ...

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: Optional[int] = 3600,
        **kwargs
    ) -> LLMResponse:
        # Generate text
        # ...

        return LLMResponse(
            text="Generated text",
            model_uri=str(self.model_uri),
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            metadata={
                "model": "my-model",
                "provider": "my-provider",
            },
            function_call=None,  # Add if the model calls a function
            tool_calls=None,     # Add if the model calls tools
        )

    async def generate_streaming(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        # Generate text with streaming
        # ...

        yield "Generated "
        yield "text"

        # For function calls, yield a dictionary
        # yield {"function_call": {"name": "function_name", "arguments": "{}"}}

    def get_metadata(self) -> ModelMetadata:
        # Return metadata about the model
        # ...

        return ModelMetadata(
            name="my-model",
            provider="my-provider",
            version="1.0",
            description="My custom model",
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                # Add other capabilities as needed:
                # ModelCapability.FUNCTION_CALLING,
                # ModelCapability.JSON_MODE,
                # ModelCapability.CODE_GENERATION,
            ],
            roles=[ModelRole.GENERAL],
            context_window=4096,
            max_tokens_per_request=1024,
            cost_per_1k_tokens_input=0.0,
            cost_per_1k_tokens_output=0.0,
        )

    def estimate_tokens(self, text: str) -> int:
        # Estimate the number of tokens in a text
        # ...

        return len(text.split())

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # Estimate the cost of a request
        # ...

        return 0.0

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        json_mode: bool = False,
        use_cache: bool = False,
        cache_namespace: str = "default",
        cache_ttl: Optional[int] = 3600,
        **kwargs
    ) -> LLMResponse:
        # Generate a response to a conversation
        # This is a convenience method that calls generate with the messages
        return await self.generate(
            prompt=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            functions=functions,
            function_call=function_call,
            json_mode=json_mode,
            use_cache=use_cache,
            cache_namespace=cache_namespace,
            cache_ttl=cache_ttl,
            **kwargs
        )
```

You can then register your adapter as a plugin:

```python
from saplings.core.plugin import ModelAdapterPlugin

# Make your adapter a plugin by inheriting from ModelAdapterPlugin
class MyCustomAdapter(LLM, ModelAdapterPlugin):
    # Implementation as shown above
    pass
```

## Conclusion

The model adapter system in Saplings provides a flexible way to use different LLM providers for inference. By using the `LLM.from_uri` method, you can easily switch between providers without changing your code.

For more information, see the [API reference](./api_reference.md) and the [examples](../examples) directory.
