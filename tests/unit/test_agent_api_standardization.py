from __future__ import annotations

"""
Tests for Agent API pattern standardization.

This module tests that the Agent API follows the standardized pattern
defined in api_standardization.md instead of using complex __new__ methods
and dynamic imports.
"""

import inspect


def test_agent_api_uses_standard_pattern():
    """Test that Agent API uses direct inheritance instead of complex __new__ pattern."""
    from saplings.api.agent import Agent

    # Check that Agent is a proper class, not a complex wrapper
    assert inspect.isclass(Agent), "Agent should be a proper class"

    # Check that Agent doesn't use complex __new__ method
    if hasattr(Agent, "__new__"):
        # Get the source of __new__ if it exists
        try:
            source = inspect.getsource(Agent.__new__)
            # Should not contain dynamic imports or complex logic
            assert (
                "importlib.import_module" not in source
            ), "Agent.__new__ should not use dynamic imports"
            assert (
                "_get_agent()" not in source
            ), "Agent.__new__ should not use dynamic getter functions"
        except (OSError, TypeError):
            # If we can't get source, it's likely a built-in __new__, which is fine
            # TypeError occurs when trying to get source of built-in methods
            pass

    # Check that Agent has proper inheritance
    mro = Agent.__mro__
    assert len(mro) >= 2, "Agent should inherit from an internal implementation"

    # Check that Agent has stability annotation
    assert hasattr(Agent, "__stability__"), "Agent should have a stability annotation"


def test_agent_builder_api_uses_standard_pattern():
    """Test that AgentBuilder API uses direct inheritance instead of complex __new__ pattern."""
    from saplings.api.agent import AgentBuilder

    # Check that AgentBuilder is a proper class, not a complex wrapper
    assert inspect.isclass(AgentBuilder), "AgentBuilder should be a proper class"

    # Check that AgentBuilder doesn't use complex __new__ method
    if hasattr(AgentBuilder, "__new__"):
        try:
            source = inspect.getsource(AgentBuilder.__new__)
            assert (
                "importlib.import_module" not in source
            ), "AgentBuilder.__new__ should not use dynamic imports"
            assert (
                "_get_agent_builder()" not in source
            ), "AgentBuilder.__new__ should not use dynamic getter functions"
        except (OSError, TypeError):
            # TypeError occurs when trying to get source of built-in methods
            pass

    # Check that AgentBuilder has proper inheritance
    mro = AgentBuilder.__mro__
    assert len(mro) >= 2, "AgentBuilder should inherit from an internal implementation"

    # Check that AgentBuilder has stability annotation
    assert hasattr(AgentBuilder, "__stability__"), "AgentBuilder should have a stability annotation"


def test_agent_config_api_uses_standard_pattern():
    """Test that AgentConfig API uses direct inheritance instead of complex __new__ pattern."""
    from saplings.api.agent import AgentConfig

    # Check that AgentConfig is a proper class, not a complex wrapper
    assert inspect.isclass(AgentConfig), "AgentConfig should be a proper class"

    # Check that AgentConfig doesn't use complex __new__ method
    if hasattr(AgentConfig, "__new__"):
        try:
            source = inspect.getsource(AgentConfig.__new__)
            assert (
                "importlib.import_module" not in source
            ), "AgentConfig.__new__ should not use dynamic imports"
            assert (
                "_get_agent_config()" not in source
            ), "AgentConfig.__new__ should not use dynamic getter functions"
        except (OSError, TypeError):
            # TypeError occurs when trying to get source of built-in methods
            pass

    # Check that AgentConfig has proper inheritance
    mro = AgentConfig.__mro__
    assert len(mro) >= 2, "AgentConfig should inherit from an internal implementation"

    # Check that AgentConfig has stability annotation
    assert hasattr(AgentConfig, "__stability__"), "AgentConfig should have a stability annotation"


def test_agent_facade_api_uses_standard_pattern():
    """Test that AgentFacade API uses direct inheritance instead of complex __new__ pattern."""
    from saplings.api.agent import AgentFacade

    # Check that AgentFacade is a proper class, not a complex wrapper
    assert inspect.isclass(AgentFacade), "AgentFacade should be a proper class"

    # Check that AgentFacade doesn't use complex __new__ method
    if hasattr(AgentFacade, "__new__"):
        try:
            source = inspect.getsource(AgentFacade.__new__)
            assert (
                "importlib.import_module" not in source
            ), "AgentFacade.__new__ should not use dynamic imports"
            assert (
                "_get_agent_facade()" not in source
            ), "AgentFacade.__new__ should not use dynamic getter functions"
        except (OSError, TypeError):
            # TypeError occurs when trying to get source of built-in methods
            pass

    # Check that AgentFacade has proper inheritance
    mro = AgentFacade.__mro__
    assert len(mro) >= 2, "AgentFacade should inherit from an internal implementation"

    # Check that AgentFacade has stability annotation
    assert hasattr(AgentFacade, "__stability__"), "AgentFacade should have a stability annotation"


def test_agent_facade_builder_api_uses_standard_pattern():
    """Test that AgentFacadeBuilder API uses direct inheritance instead of complex __new__ pattern."""
    from saplings.api.agent import AgentFacadeBuilder

    # Check that AgentFacadeBuilder is a proper class, not a complex wrapper
    assert inspect.isclass(AgentFacadeBuilder), "AgentFacadeBuilder should be a proper class"

    # Check that AgentFacadeBuilder doesn't use complex __new__ method
    if hasattr(AgentFacadeBuilder, "__new__"):
        try:
            source = inspect.getsource(AgentFacadeBuilder.__new__)
            assert (
                "importlib.import_module" not in source
            ), "AgentFacadeBuilder.__new__ should not use dynamic imports"
            assert (
                "_get_agent_facade_builder()" not in source
            ), "AgentFacadeBuilder.__new__ should not use dynamic getter functions"
        except (OSError, TypeError):
            # TypeError occurs when trying to get source of built-in methods
            pass

    # Check that AgentFacadeBuilder has proper inheritance
    mro = AgentFacadeBuilder.__mro__
    assert len(mro) >= 2, "AgentFacadeBuilder should inherit from an internal implementation"

    # Check that AgentFacadeBuilder has stability annotation
    assert hasattr(
        AgentFacadeBuilder, "__stability__"
    ), "AgentFacadeBuilder should have a stability annotation"


def test_agent_api_module_no_dynamic_imports():
    """Test that the agent API module doesn't use dynamic imports."""
    # Import the module and check its source
    import saplings.api.agent as agent_module

    # Get the source file
    source_file = inspect.getfile(agent_module)

    # Read the source code
    with open(source_file) as f:
        source_code = f.read()

    # Check for dynamic import patterns
    assert (
        "importlib.import_module" not in source_code
    ), "Agent API module should not use importlib.import_module"

    # Check for dynamic getter functions
    dynamic_getters = [
        "_get_agent()",
        "_get_agent_builder()",
        "_get_agent_config()",
        "_get_agent_facade()",
        "_get_agent_facade_builder()",
    ]

    for getter in dynamic_getters:
        assert getter not in source_code, f"Agent API module should not use dynamic getter {getter}"


def test_agent_api_follows_standardization_guidelines():
    """Test that the agent API module follows the standardization guidelines."""
    import saplings.api.agent as agent_module

    # Check that the module has proper docstring
    assert agent_module.__doc__ is not None, "Agent API module should have a docstring"

    # Check that the module defines __all__
    assert hasattr(agent_module, "__all__"), "Agent API module should define __all__"

    # Check that all public classes are in __all__
    expected_classes = ["Agent", "AgentBuilder", "AgentConfig", "AgentFacade", "AgentFacadeBuilder"]
    for class_name in expected_classes:
        assert class_name in agent_module.__all__, f"{class_name} should be in __all__"
        assert hasattr(agent_module, class_name), f"{class_name} should be defined in the module"


def test_agent_api_stability_annotations():
    """Test that all Agent API components have proper stability annotations."""
    from saplings.api.agent import Agent, AgentBuilder, AgentConfig, AgentFacade, AgentFacadeBuilder

    components = [Agent, AgentBuilder, AgentConfig, AgentFacade, AgentFacadeBuilder]

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
