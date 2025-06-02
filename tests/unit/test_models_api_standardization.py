from __future__ import annotations

"""
Tests for Models API pattern standardization.

This module tests that the Models API follows the standardized pattern
defined in api_standardization.md using direct inheritance with stability
annotations for classes and decorator wrapping for enums.
"""

import inspect


def test_models_api_uses_standard_pattern():
    """Test that Models API uses direct inheritance for classes."""
    from saplings.api.models import LLM

    # Check that LLM is a proper class, not a complex wrapper
    assert inspect.isclass(LLM), "LLM should be a proper class"

    # Check that LLM doesn't use complex __new__ method
    if hasattr(LLM, "__new__"):
        # Get the source of __new__ if it exists
        try:
            source = inspect.getsource(LLM.__new__)
            # Should not contain dynamic imports or complex logic
            assert (
                "importlib.import_module" not in source
            ), "LLM.__new__ should not use dynamic imports"
            assert "_get_llm()" not in source, "LLM.__new__ should not use dynamic getter functions"
        except (OSError, TypeError):
            # If we can't get source, it's likely a built-in __new__, which is fine
            # TypeError occurs when trying to get source of built-in methods
            pass

    # Check that LLM has proper inheritance
    mro = LLM.__mro__
    assert len(mro) >= 2, "LLM should inherit from an internal implementation"

    # Check that LLM has stability annotation
    assert hasattr(LLM, "__stability__"), "LLM should have a stability annotation"


def test_llm_builder_api_uses_standard_pattern():
    """Test that LLMBuilder API uses direct inheritance."""
    from saplings.api.models import LLMBuilder

    # Check that LLMBuilder is a proper class
    assert inspect.isclass(LLMBuilder), "LLMBuilder should be a proper class"

    # Check inheritance
    mro = LLMBuilder.__mro__
    assert len(mro) >= 2, "LLMBuilder should inherit from an internal implementation"

    # Check stability annotation
    assert hasattr(LLMBuilder, "__stability__"), "LLMBuilder should have a stability annotation"


def test_llm_response_api_uses_standard_pattern():
    """Test that LLMResponse API uses direct inheritance."""
    from saplings.api.models import LLMResponse

    # Check that LLMResponse is a proper class
    assert inspect.isclass(LLMResponse), "LLMResponse should be a proper class"

    # Check inheritance
    mro = LLMResponse.__mro__
    assert len(mro) >= 2, "LLMResponse should inherit from an internal implementation"

    # Check stability annotation
    assert hasattr(LLMResponse, "__stability__"), "LLMResponse should have a stability annotation"


def test_model_metadata_api_uses_standard_pattern():
    """Test that ModelMetadata API uses direct inheritance."""
    from saplings.api.models import ModelMetadata

    # Check that ModelMetadata is a proper class
    assert inspect.isclass(ModelMetadata), "ModelMetadata should be a proper class"

    # Check inheritance
    mro = ModelMetadata.__mro__
    assert len(mro) >= 2, "ModelMetadata should inherit from an internal implementation"

    # Check stability annotation
    assert hasattr(
        ModelMetadata, "__stability__"
    ), "ModelMetadata should have a stability annotation"


def test_model_enums_use_decorator_pattern():
    """Test that model enums use decorator wrapping pattern."""
    from saplings.api.models import ModelCapability, ModelRole

    # Check that enums have stability annotations
    assert hasattr(
        ModelCapability, "__stability__"
    ), "ModelCapability should have a stability annotation"
    assert hasattr(ModelRole, "__stability__"), "ModelRole should have a stability annotation"

    # Check that they are still proper enums
    from enum import Enum

    assert issubclass(ModelCapability, Enum), "ModelCapability should be an Enum"
    assert issubclass(ModelRole, Enum), "ModelRole should be an Enum"


def test_models_api_module_no_dynamic_imports():
    """Test that the models API module doesn't use dynamic imports."""
    # Import the module and check its source
    import saplings.api.models as models_module

    # Get the source file
    source_file = inspect.getfile(models_module)

    # Read the source code
    with open(source_file) as f:
        source_code = f.read()

    # Check for dynamic import patterns
    assert (
        "importlib.import_module" not in source_code
    ), "Models API module should not use importlib.import_module"

    # Check for dynamic getter functions
    dynamic_getters = [
        "_get_llm()",
        "_get_llm_builder()",
        "_get_llm_response()",
        "_get_model_metadata()",
    ]

    for getter in dynamic_getters:
        assert (
            getter not in source_code
        ), f"Models API module should not use dynamic getter {getter}"


def test_models_api_follows_standardization_guidelines():
    """Test that the models API module follows the standardization guidelines."""
    import saplings.api.models as models_module

    # Check that the module defines __all__
    assert hasattr(models_module, "__all__"), "Models API module should define __all__"

    # Check that all public classes are in __all__
    expected_classes = ["LLM", "LLMBuilder", "LLMResponse", "ModelMetadata"]

    for class_name in expected_classes:
        assert class_name in models_module.__all__, f"{class_name} should be in __all__"
        assert hasattr(models_module, class_name), f"{class_name} should be defined in the module"

    # Check that enums are in __all__
    expected_enums = ["ModelCapability", "ModelRole"]
    for enum_name in expected_enums:
        assert enum_name in models_module.__all__, f"{enum_name} should be in __all__"
        assert hasattr(models_module, enum_name), f"{enum_name} should be defined in the module"


def test_models_api_stability_annotations():
    """Test that all Models API components have proper stability annotations."""
    from saplings.api.models import (
        LLM,
        LLMBuilder,
        LLMResponse,
        ModelCapability,
        ModelMetadata,
        ModelRole,
    )

    components = [LLM, LLMBuilder, LLMResponse, ModelMetadata, ModelCapability, ModelRole]

    for component in components:
        assert hasattr(
            component, "__stability__"
        ), f"{component.__name__} should have a stability annotation"

        stability = component.__stability__
        assert stability in [
            "stable",
            "beta",
            "alpha",
        ], f"{component.__name__} should have a valid stability level"


def test_models_api_uses_direct_inheritance():
    """Test that Models API classes use direct inheritance pattern."""
    from saplings.api.models import LLM, LLMBuilder, LLMResponse, ModelMetadata

    classes = [LLM, LLMBuilder, LLMResponse, ModelMetadata]

    for cls in classes:
        # Check inheritance
        assert cls.__bases__, f"{cls.__name__} should have base classes"
        # Should inherit directly from internal implementation
        base_module = cls.__bases__[0].__module__
        assert (
            "saplings.models._internal" in base_module
        ), f"{cls.__name__} should inherit from internal implementation"


def test_adapter_imports_available():
    """Test that adapter imports are available through the models API."""
    from saplings.api.models import (
        AnthropicAdapter,
        HuggingFaceAdapter,
        OpenAIAdapter,
        VLLMAdapter,
    )

    adapters = [AnthropicAdapter, HuggingFaceAdapter, OpenAIAdapter, VLLMAdapter]

    for adapter in adapters:
        assert inspect.isclass(adapter), f"{adapter.__name__} should be a class"
        # Adapters should have stability annotations (they come from adapters API)
        assert hasattr(
            adapter, "__stability__"
        ), f"{adapter.__name__} should have a stability annotation"


def test_models_api_mixed_pattern_consistency():
    """Test that the mixed pattern (inheritance for classes, decorator for enums) is consistent."""
    from saplings.api.models import LLM, ModelCapability

    # Classes should use inheritance
    assert len(LLM.__mro__) >= 2, "Classes should use inheritance pattern"

    # Enums should use decorator pattern (they should still be the original enum type)
    from enum import Enum

    assert issubclass(ModelCapability, Enum), "Enums should still be proper Enum types"

    # Both should have stability annotations
    assert hasattr(LLM, "__stability__"), "Classes should have stability annotations"
    assert hasattr(ModelCapability, "__stability__"), "Enums should have stability annotations"
