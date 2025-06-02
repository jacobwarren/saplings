from __future__ import annotations

"""
Models API module for Saplings.

This module provides the public API for model adapters and LLM interfaces.
"""

# Import directly from models component
# Import lightweight adapters from the public API
from saplings.api.adapters import (
    AnthropicAdapter,
    OpenAIAdapter,
)
from saplings.models._internal.interfaces import (
    LLM as _LLM,
)
from saplings.models._internal.interfaces import (
    LLMResponse as _LLMResponse,
)
from saplings.models._internal.interfaces import (
    ModelCapability,
    ModelRole,
)
from saplings.models._internal.interfaces import (
    ModelMetadata as _ModelMetadata,
)
from saplings.models._internal.llm_builder import LLMBuilder as _LLMBuilder


# Lazy import heavy adapters
def _get_heavy_adapter(name: str):
    """Get heavy adapter using lazy import."""
    from saplings.api.adapters import __getattr__ as adapters_getattr

    return adapters_getattr(name)


# Import stability decorators
from saplings.api.stability import stable


# Re-export the model classes with stability annotations
@stable
class LLM(_LLM):
    """
    Base class for language models.

    This class defines the interface for all language models in the Saplings framework.
    It provides methods for generating text and managing model state.
    """


@stable
class LLMBuilder(_LLMBuilder):
    """
    Builder for language models.

    This class provides a fluent interface for configuring and creating language models.
    """


@stable
class LLMResponse(_LLMResponse):
    """
    Response from a language model.

    This class encapsulates the response from a language model, including the generated
    text, token counts, and metadata.
    """


@stable
class ModelMetadata(_ModelMetadata):
    """
    Metadata for a language model.

    This class stores metadata about a language model, including its capabilities,
    token limits, and other attributes.
    """


# Add stability annotations to the enums
ModelCapability = stable(ModelCapability)
ModelRole = stable(ModelRole)


# Lazy import heavy adapters
def __getattr__(name: str):
    """Lazy import heavy adapters using centralized system."""
    if name in ["HuggingFaceAdapter", "TransformersAdapter", "VLLMAdapter", "VLLMFallbackAdapter"]:
        return _get_heavy_adapter(name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Define the public API
__all__ = [
    # Model classes
    "LLM",
    "LLMBuilder",
    "LLMResponse",
    "ModelMetadata",
    # Enums
    "ModelCapability",
    "ModelRole",
    # Adapters
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "OpenAIAdapter",
    "TransformersAdapter",
    "VLLMAdapter",
    "VLLMFallbackAdapter",
]
