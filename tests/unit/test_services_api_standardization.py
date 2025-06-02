from __future__ import annotations

"""
Tests for Services API pattern standardization.

This module tests that the Services API follows the standardized pattern
defined in api_standardization.md using direct inheritance with stability
annotations and minimal lazy loading only where necessary.
"""

import inspect


def test_services_api_uses_standard_pattern():
    """Test that Services API uses direct inheritance for service classes."""
    from saplings.api.services import ExecutionService

    # Check that ExecutionService is a proper class, not a complex wrapper
    assert inspect.isclass(ExecutionService), "ExecutionService should be a proper class"

    # Check that ExecutionService doesn't use complex __new__ method
    if hasattr(ExecutionService, "__new__"):
        # Get the source of __new__ if it exists
        try:
            source = inspect.getsource(ExecutionService.__new__)
            # Should not contain dynamic imports or complex logic
            assert (
                "importlib.import_module" not in source
            ), "ExecutionService.__new__ should not use dynamic imports"
            assert (
                "_get_execution_service()" not in source
            ), "ExecutionService.__new__ should not use dynamic getter functions"
        except (OSError, TypeError):
            # If we can't get source, it's likely a built-in __new__, which is fine
            # TypeError occurs when trying to get source of built-in methods
            pass

    # Check that ExecutionService has proper inheritance
    mro = ExecutionService.__mro__
    assert len(mro) >= 2, "ExecutionService should inherit from an internal implementation"

    # Check that ExecutionService has stability annotation
    assert hasattr(
        ExecutionService, "__stability__"
    ), "ExecutionService should have a stability annotation"


def test_service_classes_use_standard_pattern():
    """Test that all service classes use direct inheritance."""
    from saplings.api.services import (
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
    )

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
        # Check that it's a proper class
        assert inspect.isclass(service_class), f"{service_class.__name__} should be a proper class"

        # Check inheritance
        mro = service_class.__mro__
        assert (
            len(mro) >= 2
        ), f"{service_class.__name__} should inherit from an internal implementation"

        # Check stability annotation
        assert hasattr(
            service_class, "__stability__"
        ), f"{service_class.__name__} should have a stability annotation"


def test_service_builders_available():
    """Test that service builders are available (through lazy loading if necessary)."""
    from saplings.api.services import (
        ExecutionServiceBuilder,
        JudgeServiceBuilder,
        MemoryManagerBuilder,
        ModalityServiceBuilder,
        OrchestrationServiceBuilder,
        PlannerServiceBuilder,
        RetrievalServiceBuilder,
        SelfHealingServiceBuilder,
        ToolServiceBuilder,
        ValidatorServiceBuilder,
    )

    builders = [
        ExecutionServiceBuilder,
        JudgeServiceBuilder,
        MemoryManagerBuilder,
        ModalityServiceBuilder,
        OrchestrationServiceBuilder,
        PlannerServiceBuilder,
        RetrievalServiceBuilder,
        SelfHealingServiceBuilder,
        ToolServiceBuilder,
        ValidatorServiceBuilder,
    ]

    for builder_class in builders:
        # Check that it's a proper class
        assert inspect.isclass(builder_class), f"{builder_class.__name__} should be a proper class"

        # Check stability annotation
        assert hasattr(
            builder_class, "__stability__"
        ), f"{builder_class.__name__} should have a stability annotation"


def test_services_api_follows_standardization_guidelines():
    """Test that the services API module follows the standardization guidelines."""
    import saplings.api.services as services_module

    # Check that the module defines __all__
    assert hasattr(services_module, "__all__"), "Services API module should define __all__"

    # Check that all public service classes are in __all__
    expected_services = [
        "ExecutionService",
        "JudgeService",
        "MemoryManager",
        "ModalityService",
        "OrchestrationService",
        "PlannerService",
        "RetrievalService",
        "SelfHealingService",
        "ToolService",
        "ValidatorService",
    ]

    for service_name in expected_services:
        assert service_name in services_module.__all__, f"{service_name} should be in __all__"
        assert hasattr(
            services_module, service_name
        ), f"{service_name} should be defined in the module"


def test_services_api_stability_annotations():
    """Test that all Services API components have proper stability annotations."""
    from saplings.api.services import (
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
    )

    components = [
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


def test_services_api_uses_direct_inheritance():
    """Test that Services API classes use direct inheritance pattern."""
    from saplings.api.services import ExecutionService, JudgeService, MemoryManager

    services = [ExecutionService, JudgeService, MemoryManager]

    for service_class in services:
        # Check inheritance
        assert service_class.__bases__, f"{service_class.__name__} should have base classes"
        # Should inherit directly from internal implementation
        base_module = service_class.__bases__[0].__module__
        assert (
            "saplings._internal.services" in base_module
            or "saplings.api.service_impl" in base_module
            or "saplings.services._internal" in base_module
        ), f"{service_class.__name__} should inherit from internal implementation, got {base_module}"


def test_services_api_lazy_loading_minimal():
    """Test that lazy loading is used minimally and only where necessary."""
    import saplings.api.services as services_module

    # Check if the module has __getattr__ (indicates lazy loading)
    has_getattr = hasattr(services_module, "__getattr__")

    if has_getattr:
        # If lazy loading is used, it should be documented and minimal
        # Check that it's only used for builders (to avoid circular imports)
        import inspect

        source_file = inspect.getfile(services_module)

        with open(source_file) as f:
            source_code = f.read()

        # Should mention builders or circular imports in the lazy loading section
        assert (
            "builder" in source_code.lower() or "circular" in source_code.lower()
        ), "Lazy loading should only be used for builders or to avoid circular imports"


def test_services_api_no_unnecessary_dynamic_imports():
    """Test that the services API module doesn't use unnecessary dynamic imports."""
    import saplings.api.services as services_module

    # Get the source file
    source_file = inspect.getfile(services_module)

    # Read the source code
    with open(source_file) as f:
        source_code = f.read()

    # Count dynamic import patterns
    dynamic_import_count = source_code.count("importlib.import_module")

    # Should have minimal dynamic imports (only for necessary lazy loading)
    assert dynamic_import_count <= 1, "Services API module should have minimal dynamic imports"


def test_base_service_class_available():
    """Test that the base Service class is available and properly structured."""
    from saplings.api.services import Service

    # Check that Service is a proper class
    assert inspect.isclass(Service), "Service should be a proper class"

    # Check stability annotation
    assert hasattr(Service, "__stability__"), "Service should have a stability annotation"

    # Check that it has basic service functionality
    assert hasattr(Service, "__init__"), "Service should have __init__ method"
    assert hasattr(Service, "name"), "Service should have name property"


def test_services_api_builder_pattern_consistency():
    """Test that service builders follow a consistent pattern."""
    from saplings.api.services import ExecutionServiceBuilder, JudgeServiceBuilder

    builders = [ExecutionServiceBuilder, JudgeServiceBuilder]

    for builder_class in builders:
        # Check that it's a proper class
        assert inspect.isclass(builder_class), f"{builder_class.__name__} should be a proper class"

        # Check stability annotation
        assert hasattr(
            builder_class, "__stability__"
        ), f"{builder_class.__name__} should have a stability annotation"

        # Builder classes should typically have a build method
        # (This is a pattern check, not a strict requirement)
        if hasattr(builder_class, "build"):
            assert callable(
                builder_class.build
            ), f"{builder_class.__name__}.build should be callable"
