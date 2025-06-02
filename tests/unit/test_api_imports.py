from __future__ import annotations

"""
Tests for API imports in the Saplings library.

This module contains tests to verify that modules import from the public API.
"""

import importlib


def test_module_imports_from_public_api(module_name: str, api_module: str) -> None:
    """Test that a module imports from the public API.

    Args:
        module_name: The name of the module to check
        api_module: The name of the API module it should import from
    """
    # Import the module
    module = importlib.import_module(module_name)

    # Check the source code of the module

    # Get the source file
    source_file = inspect.getfile(module)

    # Read the source code
    with open(source_file) as f:
        source_code = f.read()

    # Check that the module imports from the public API
    assert (
        f"from saplings.api.{api_module} import" in source_code
    ), f"The {module_name} module should import from saplings.api.{api_module}"

    # Check that it doesn't import directly from _internal
    assert (
        f"from {module_name}._internal" not in source_code
    ), f"The {module_name} module should not import directly from _internal"


def test_adapters_imports_from_public_api() -> None:
    """Test that the adapters module imports from the public API."""
    test_module_imports_from_public_api("saplings.adapters", "models")


def test_tools_imports_from_public_api() -> None:
    """Test that the tools module imports from the public API."""
    test_module_imports_from_public_api("saplings.tools", "tools")


def test_memory_imports_from_public_api() -> None:
    """Test that the memory module imports from the public API."""
    test_module_imports_from_public_api("saplings.memory", "memory")


def test_services_imports_from_public_api() -> None:
    """Test that the services module imports from the public API."""
    test_module_imports_from_public_api("saplings.services", "services")


def test_validator_imports_from_public_api() -> None:
    """Test that the validator module imports from the public API."""
    test_module_imports_from_public_api("saplings.validator", "validator")


def test_retrieval_imports_from_public_api() -> None:
    """Test that the retrieval module imports from the public API."""
    test_module_imports_from_public_api("saplings.retrieval", "retrieval")


def test_monitoring_imports_from_public_api() -> None:
    """Test that the monitoring module imports from the public API."""
    test_module_imports_from_public_api("saplings.monitoring", "monitoring")


def test_judge_imports_from_public_api() -> None:
    """Test that the judge module imports from the public API."""
    test_module_imports_from_public_api("saplings.judge", "judge")
