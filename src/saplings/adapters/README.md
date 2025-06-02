# Saplings Model Adapters

This package provides adapter implementations for various LLM providers in the Saplings framework.

## API Structure

The model adapters follow the Saplings API separation pattern:

1. **Public API**: Exposed through `saplings.api.models` and re-exported at the top level of the `saplings` package
2. **Internal Implementation**: Located in the `_internal` directory

## Usage

To use the model adapters, import them from the public API:

```python
# Recommended: Import from the top-level package
from saplings import OpenAIAdapter, AnthropicAdapter, VLLMAdapter

# Alternative: Import directly from the API module
from saplings.api.models import OpenAIAdapter, AnthropicAdapter, VLLMAdapter
```

Do not import directly from the internal implementation:

```python
# Don't do this
from saplings.adapters._internal.openai_adapter import OpenAIAdapter  # Wrong
```

## Available Adapters

The following model adapters are available:

- `OpenAIAdapter`: Adapter for OpenAI models
- `AnthropicAdapter`: Adapter for Anthropic models
- `HuggingFaceAdapter`: Adapter for Hugging Face models
- `VLLMAdapter`: Adapter for vLLM models

## Implementation Details

The adapter implementations are located in the `_internal` directory:

- `_internal/openai_adapter.py`: Implementation of the OpenAI adapter
- `_internal/anthropic_adapter.py`: Implementation of the Anthropic adapter
- `_internal/huggingface_adapter.py`: Implementation of the Hugging Face adapter
- `_internal/vllm_adapter.py`: Implementation of the vLLM adapter
- `_internal/transformers_adapter.py`: Implementation of the Transformers adapter
- `_internal/vllm_fallback_adapter.py`: Fallback implementation for vLLM

These internal implementations are wrapped by the public API in `saplings.api.models` to provide stability annotations and a consistent interface.

## Entry Points

The model adapters are registered as entry points in `pyproject.toml`:

```toml
[project.entry-points."saplings.model_adapters"]
vllm = "saplings.api.models:VLLMAdapter"
openai = "saplings.api.models:OpenAIAdapter"
anthropic = "saplings.api.models:AnthropicAdapter"
huggingface = "saplings.api.models:HuggingFaceAdapter"
```

These entry points allow the adapters to be discovered and registered by the plugin system.

## Contributing

When contributing to the model adapters, follow these guidelines:

1. Add new adapter implementations to the `_internal` directory
2. Add public API wrappers to `saplings.api.models`
3. Add stability annotations to all public API components
4. Document all public API components with docstrings
5. Update entry points to point to the public API components
6. Ensure the top-level `__init__.py` imports from the public API, not internal modules
