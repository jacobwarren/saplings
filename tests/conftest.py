from __future__ import annotations

"""Test fixtures for Saplings tests."""


import os
from pathlib import Path

import pytest

from saplings.agent_config import AgentConfig
from saplings.container_config import configure_container
from saplings.di import reset_container


@pytest.fixture(autouse=True)
def reset_di():
    """Reset the DI container after each test.

    This ensures test isolation by creating a fresh container
    for each test.
    """
    # Reset before test to start clean
    reset_container()
    yield
    # Reset after test to clean up
    reset_container()


@pytest.fixture()
def test_config():
    """Create a test configuration.

    Returns:
        AgentConfig: Test configuration
    """
    test_dir = Path(os.getenv("PYTEST_TMPDIR", "/tmp/saplings_tests"))
    os.makedirs(test_dir, exist_ok=True)

    return AgentConfig(
        provider="test",
        model_name="model",
        memory_path=str(test_dir / "memory"),
        output_dir=str(test_dir / "output"),
        enable_monitoring=False,
        enable_self_healing=False,
        enable_tool_factory=False,
    )


@pytest.fixture()
def test_container(test_config):
    """Create a test container with test configuration.

    Args:
        test_config: Test configuration from the test_config fixture

    Returns:
        Container: Configured test container
    """
    return configure_container(test_config)
