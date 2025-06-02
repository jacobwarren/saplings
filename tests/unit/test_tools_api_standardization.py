from __future__ import annotations

"""
Tests for Tools API pattern standardization.

This module tests that the Tools API follows the standardized pattern
defined in api_standardization.md using direct inheritance with stability
annotations instead of complex patterns.
"""

import inspect


def test_tools_api_uses_standard_pattern():
    """Test that Tools API uses direct inheritance instead of complex patterns."""
    from saplings.api.tools import Tool

    # Check that Tool is a proper class, not a complex wrapper
    assert inspect.isclass(Tool), "Tool should be a proper class"

    # Check that Tool doesn't use complex __new__ method
    if hasattr(Tool, "__new__"):
        # Get the source of __new__ if it exists
        try:
            source = inspect.getsource(Tool.__new__)
            # Should not contain dynamic imports or complex logic
            assert (
                "importlib.import_module" not in source
            ), "Tool.__new__ should not use dynamic imports"
            assert (
                "_get_tool()" not in source
            ), "Tool.__new__ should not use dynamic getter functions"
        except (OSError, TypeError):
            # If we can't get source, it's likely a built-in __new__, which is fine
            # TypeError occurs when trying to get source of built-in methods
            pass

    # Check that Tool has proper inheritance
    mro = Tool.__mro__
    assert len(mro) >= 2, "Tool should inherit from an internal implementation"

    # Check that Tool has stability annotation
    assert hasattr(Tool, "__stability__"), "Tool should have a stability annotation"


def test_tool_collection_api_uses_standard_pattern():
    """Test that ToolCollection API uses direct inheritance."""
    from saplings.api.tools import ToolCollection

    # Check that ToolCollection is a proper class
    assert inspect.isclass(ToolCollection), "ToolCollection should be a proper class"

    # Check inheritance
    mro = ToolCollection.__mro__
    assert len(mro) >= 2, "ToolCollection should inherit from an internal implementation"

    # Check stability annotation
    assert hasattr(
        ToolCollection, "__stability__"
    ), "ToolCollection should have a stability annotation"


def test_tool_registry_api_uses_standard_pattern():
    """Test that ToolRegistry API uses direct inheritance."""
    from saplings.api.tools import ToolRegistry

    # Check that ToolRegistry is a proper class
    assert inspect.isclass(ToolRegistry), "ToolRegistry should be a proper class"

    # Check inheritance
    mro = ToolRegistry.__mro__
    assert len(mro) >= 2, "ToolRegistry should inherit from an internal implementation"

    # Check stability annotation
    assert hasattr(ToolRegistry, "__stability__"), "ToolRegistry should have a stability annotation"


def test_default_tools_use_standard_pattern():
    """Test that default tool classes use direct inheritance."""
    from saplings.api.tools import (
        DuckDuckGoSearchTool,
        FinalAnswerTool,
        GoogleSearchTool,
        PythonInterpreterTool,
        UserInputTool,
        VisitWebpageTool,
        WikipediaSearchTool,
    )

    tools = [
        DuckDuckGoSearchTool,
        FinalAnswerTool,
        GoogleSearchTool,
        PythonInterpreterTool,
        UserInputTool,
        VisitWebpageTool,
        WikipediaSearchTool,
    ]

    for tool_class in tools:
        # Check that it's a proper class
        assert inspect.isclass(tool_class), f"{tool_class.__name__} should be a proper class"

        # Check inheritance
        mro = tool_class.__mro__
        assert (
            len(mro) >= 2
        ), f"{tool_class.__name__} should inherit from an internal implementation"

        # Check stability annotation
        assert hasattr(
            tool_class, "__stability__"
        ), f"{tool_class.__name__} should have a stability annotation"


def test_tool_functions_have_stability_annotations():
    """Test that tool functions have proper stability annotations."""
    from saplings.api.tools import (
        get_all_default_tools,
        get_default_tool,
        get_registered_tools,
        is_browser_tools_available,
        is_mcp_available,
        register_tool,
        tool,
        validate_tool,
        validate_tool_attributes,
        validate_tool_parameters,
    )

    functions = [
        get_all_default_tools,
        get_default_tool,
        get_registered_tools,
        is_browser_tools_available,
        is_mcp_available,
        register_tool,
        tool,
        validate_tool,
        validate_tool_attributes,
        validate_tool_parameters,
    ]

    for func in functions:
        assert hasattr(func, "__stability__"), f"{func.__name__} should have a stability annotation"

        stability = func.__stability__
        assert stability in [
            "stable",
            "beta",
            "alpha",
        ], f"{func.__name__} should have a valid stability level"


def test_tools_api_module_no_dynamic_imports():
    """Test that the tools API module doesn't use dynamic imports."""
    # Import the module and check its source
    import saplings.api.tools as tools_module

    # Get the source file
    source_file = inspect.getfile(tools_module)

    # Read the source code
    with open(source_file) as f:
        source_code = f.read()

    # Check for dynamic import patterns
    assert (
        "importlib.import_module" not in source_code
    ), "Tools API module should not use importlib.import_module"

    # Check for dynamic getter functions
    dynamic_getters = [
        "_get_tool()",
        "_get_tool_collection()",
        "_get_tool_registry()",
    ]

    for getter in dynamic_getters:
        assert getter not in source_code, f"Tools API module should not use dynamic getter {getter}"


def test_tools_api_follows_standardization_guidelines():
    """Test that the tools API module follows the standardization guidelines."""
    import saplings.api.tools as tools_module

    # Check that the module defines __all__
    assert hasattr(tools_module, "__all__"), "Tools API module should define __all__"

    # Check that all public classes are in __all__
    expected_classes = [
        "Tool",
        "ToolCollection",
        "ToolRegistry",
        "DuckDuckGoSearchTool",
        "FinalAnswerTool",
        "GoogleSearchTool",
        "PythonInterpreterTool",
        "UserInputTool",
        "VisitWebpageTool",
        "WikipediaSearchTool",
    ]

    for class_name in expected_classes:
        assert class_name in tools_module.__all__, f"{class_name} should be in __all__"
        assert hasattr(tools_module, class_name), f"{class_name} should be defined in the module"


def test_tools_api_stability_annotations():
    """Test that all Tools API components have proper stability annotations."""
    from saplings.api.tools import (
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
    )

    components = [
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


def test_tools_api_uses_direct_inheritance():
    """Test that Tools API classes use direct inheritance pattern."""
    from saplings.api.tools import Tool, ToolCollection, ToolRegistry

    # Check Tool inheritance
    assert Tool.__bases__, "Tool should have base classes"
    # Should inherit directly from internal implementation
    base_module = Tool.__bases__[0].__module__
    assert (
        "saplings.tools._internal" in base_module
    ), "Tool should inherit from internal implementation"

    # Check ToolCollection inheritance
    assert ToolCollection.__bases__, "ToolCollection should have base classes"
    base_module = ToolCollection.__bases__[0].__module__
    assert (
        "saplings.tools._internal" in base_module
    ), "ToolCollection should inherit from internal implementation"

    # Check ToolRegistry inheritance
    assert ToolRegistry.__bases__, "ToolRegistry should have base classes"
    base_module = ToolRegistry.__bases__[0].__module__
    assert (
        "saplings.tools._internal" in base_module
    ), "ToolRegistry should inherit from internal implementation"
