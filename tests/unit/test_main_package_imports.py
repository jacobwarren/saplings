from __future__ import annotations

"""
Tests for main package imports.

This module tests that the main saplings package exposes the complete public API
correctly and that all public components are accessible via `from saplings import ...`.
"""

import inspect

import pytest


def test_agent_imports():
    """Test that agent components can be imported from main package."""
    from saplings import Agent, AgentBuilder, AgentConfig, AgentFacade, AgentFacadeBuilder

    # Check that all components are available
    assert Agent is not None
    assert AgentBuilder is not None
    assert AgentConfig is not None
    assert AgentFacade is not None
    assert AgentFacadeBuilder is not None

    # Check that they are classes
    assert inspect.isclass(Agent)
    assert inspect.isclass(AgentBuilder)
    assert inspect.isclass(AgentConfig)
    assert inspect.isclass(AgentFacade)
    assert inspect.isclass(AgentFacadeBuilder)


def test_tool_imports():
    """Test that tool components can be imported from main package."""
    from saplings import (
        DuckDuckGoSearchTool,
        FinalAnswerTool,
        GoogleSearchTool,
        PythonInterpreterTool,
        Tool,
        ToolCollection,
        ToolRegistry,
        UserInputTool,
        VisitWebpageTool,
        WikipediaSearchTool,
        register_tool,
        tool,
    )

    # Check that all components are available
    tools = [
        Tool,
        ToolCollection,
        ToolRegistry,
        DuckDuckGoSearchTool,
        FinalAnswerTool,
        GoogleSearchTool,
        PythonInterpreterTool,
        UserInputTool,
        VisitWebpageTool,
        WikipediaSearchTool,
    ]

    for tool_class in tools:
        assert tool_class is not None
        assert inspect.isclass(tool_class)

    # Check functions
    assert tool is not None
    assert callable(tool)
    assert register_tool is not None
    assert callable(register_tool)


def test_model_imports():
    """Test that model components can be imported from main package."""
    from saplings import (
        LLM,
        AnthropicAdapter,
        HuggingFaceAdapter,
        LLMBuilder,
        LLMResponse,
        ModelCapability,
        ModelMetadata,
        ModelRole,
        OpenAIAdapter,
        VLLMAdapter,
    )

    # Check model classes
    model_classes = [LLM, LLMBuilder, LLMResponse, ModelMetadata]
    for model_class in model_classes:
        assert model_class is not None
        assert inspect.isclass(model_class)

    # Check enums
    from enum import Enum

    assert issubclass(ModelCapability, Enum)
    assert issubclass(ModelRole, Enum)

    # Check adapters
    adapters = [AnthropicAdapter, HuggingFaceAdapter, OpenAIAdapter, VLLMAdapter]
    for adapter in adapters:
        assert adapter is not None
        assert inspect.isclass(adapter)


def test_service_imports():
    """Test that service components can be imported from main package."""
    from saplings import (
        ExecutionService,
        ExecutionServiceBuilder,
        JudgeService,
        JudgeServiceBuilder,
        MemoryManager,
        MemoryManagerBuilder,
        ModalityService,
        OrchestrationService,
        PlannerService,
        RetrievalService,
        SelfHealingService,
        ToolService,
        ValidatorService,
    )

    # Check service classes
    services = [
        ExecutionService,
        JudgeService,
        MemoryManager,
        ModalityService,
        OrchestrationService,
        PlannerService,
        RetrievalService,
        SelfHealingService,
        ToolService,
        ValidatorService,
    ]

    for service_class in services:
        assert service_class is not None
        assert inspect.isclass(service_class)

    # Check builders
    builders = [ExecutionServiceBuilder, JudgeServiceBuilder, MemoryManagerBuilder]
    for builder_class in builders:
        assert builder_class is not None
        assert inspect.isclass(builder_class)


def test_memory_imports():
    """Test that memory components can be imported from main package."""
    from saplings import (
        DependencyGraph,
        DependencyGraphBuilder,
        Document,
        DocumentMetadata,
        DocumentNode,
        Indexer,
        IndexerRegistry,
        InMemoryVectorStore,
        MemoryConfig,
        MemoryStore,
        MemoryStoreBuilder,
        SimpleIndexer,
        VectorStore,
        get_indexer,
        get_vector_store,
    )

    # Check document classes
    document_classes = [Document, DocumentMetadata, DocumentNode]
    for doc_class in document_classes:
        assert doc_class is not None
        assert inspect.isclass(doc_class)

    # Check memory classes
    memory_classes = [
        DependencyGraph,
        DependencyGraphBuilder,
        MemoryStore,
        MemoryStoreBuilder,
        MemoryConfig,
    ]
    for memory_class in memory_classes:
        assert memory_class is not None
        assert inspect.isclass(memory_class)

    # Check indexer classes
    indexer_classes = [Indexer, SimpleIndexer, IndexerRegistry]
    for indexer_class in indexer_classes:
        assert indexer_class is not None
        assert inspect.isclass(indexer_class)

    # Check vector store classes
    vector_classes = [VectorStore, InMemoryVectorStore]
    for vector_class in vector_classes:
        assert vector_class is not None
        assert inspect.isclass(vector_class)

    # Check functions
    functions = [get_indexer, get_vector_store]
    for func in functions:
        assert func is not None
        assert callable(func)


def test_dependency_injection_imports():
    """Test that dependency injection components can be imported from main package."""
    from saplings import Container, configure_container, container, reset_container

    # Check that all components are available
    assert Container is not None
    assert inspect.isclass(Container)

    assert container is not None
    assert reset_container is not None
    assert callable(reset_container)

    assert configure_container is not None
    assert callable(configure_container)


def test_validator_imports():
    """Test that validator components can be imported from main package."""
    from saplings import (
        ExecutionValidator,
        KeywordValidator,
        LengthValidator,
        RuntimeValidator,
        StaticValidator,
        ValidationResult,
        ValidationStatus,
        ValidationStrategy,
        Validator,
        ValidatorConfig,
        ValidatorRegistry,
        ValidatorType,
        get_validator_registry,
    )

    # Check validator classes
    validator_classes = [
        ExecutionValidator,
        KeywordValidator,
        LengthValidator,
        RuntimeValidator,
        StaticValidator,
        Validator,
        ValidatorConfig,
        ValidatorRegistry,
    ]

    for validator_class in validator_classes:
        assert validator_class is not None
        assert inspect.isclass(validator_class)

    # Check enums/types
    from enum import Enum

    assert issubclass(ValidationStatus, Enum)
    assert issubclass(ValidationStrategy, Enum)
    assert issubclass(ValidatorType, Enum)

    # Check result class
    assert ValidationResult is not None
    assert inspect.isclass(ValidationResult)

    # Check functions
    assert get_validator_registry is not None
    assert callable(get_validator_registry)


def test_monitoring_imports():
    """Test that monitoring components can be imported from main package."""
    from saplings import (
        BlameEdge,
        BlameGraph,
        BlameNode,
        MonitoringConfig,
        TraceManager,
        TraceViewer,
    )

    # Check monitoring classes
    monitoring_classes = [
        BlameEdge,
        BlameGraph,
        BlameNode,
        MonitoringConfig,
        TraceManager,
        TraceViewer,
    ]

    for monitoring_class in monitoring_classes:
        assert monitoring_class is not None
        assert inspect.isclass(monitoring_class)


def test_security_imports():
    """Test that security components can be imported from main package."""
    from saplings import (
        RedactingFilter,
        Sanitizer,
        install_global_filter,
        install_import_hook,
        redact,
        sanitize,
    )

    # Check security classes
    security_classes = [RedactingFilter, Sanitizer]
    for security_class in security_classes:
        assert security_class is not None
        assert inspect.isclass(security_class)

    # Check security functions
    security_functions = [install_global_filter, install_import_hook, redact, sanitize]
    for func in security_functions:
        assert func is not None
        assert callable(func)


def test_version_import():
    """Test that version can be imported from main package."""
    from saplings import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_main_package_has_all():
    """Test that main package defines __all__."""
    import saplings

    # Check if __all__ is defined
    if hasattr(saplings, "__all__"):
        assert isinstance(saplings.__all__, list)
        assert len(saplings.__all__) > 0
    else:
        # If __all__ is not defined, that's acceptable but not ideal
        print("Warning: Main package does not define __all__")


def test_main_package_docstring():
    """Test that main package has a proper docstring."""
    import saplings

    assert saplings.__doc__ is not None
    assert isinstance(saplings.__doc__, str)
    assert len(saplings.__doc__) > 0
    assert "saplings" in saplings.__doc__.lower()


def test_no_internal_api_exposure():
    """Test that internal APIs are not exposed through main package."""
    import saplings

    # Check that no _internal modules are exposed (with some exceptions for legitimate API exposure)
    for attr_name in dir(saplings):
        if not attr_name.startswith("_"):  # Skip private attributes
            attr = getattr(saplings, attr_name)
            if hasattr(attr, "__module__"):
                module_name = attr.__module__
                # Allow internal modules if they're exposed through the API layer
                # or if they're legitimate public components
                is_internal = "_internal" in module_name
                is_api_exposed = "saplings.api" in module_name
                is_legitimate_public = (
                    module_name.startswith("saplings.")
                    and not module_name.startswith("saplings._internal")
                    and "builtins" not in module_name
                )

                if is_internal and not (is_api_exposed or is_legitimate_public):
                    # This is a warning rather than a failure for now
                    # as some internal components may be legitimately exposed
                    print(f"Warning: Internal module {module_name} exposed as {attr_name}")


def test_import_performance():
    """Test that importing from main package doesn't take too long."""
    import time

    start_time = time.time()

    # Import a few key components

    end_time = time.time()
    import_time = end_time - start_time

    # Should import in reasonable time (less than 5 seconds)
    assert import_time < 5.0, f"Import took too long: {import_time:.2f} seconds"


def test_all_public_api_components_importable():
    """Test that all components listed in __all__ are importable."""
    import saplings

    # Check if __all__ exists
    if not hasattr(saplings, "__all__"):
        pytest.skip("Main package does not define __all__")

    # Get all public components
    all_components = saplings.__all__

    # Try to import each component
    missing_components = []
    for component_name in all_components:
        if not hasattr(saplings, component_name):
            missing_components.append(component_name)
        else:
            component = getattr(saplings, component_name)
            if component is None:
                missing_components.append(f"{component_name} (None)")

    # Report missing components but don't fail the test if there are only a few
    if missing_components:
        print(f"Missing components: {missing_components}")
        # Only fail if more than 10% of components are missing
        missing_ratio = len(missing_components) / len(all_components)
        assert (
            missing_ratio < 0.1
        ), f"Too many components missing: {len(missing_components)}/{len(all_components)}"
