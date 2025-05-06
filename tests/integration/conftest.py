from __future__ import annotations

"""
Pytest configuration for integration tests.

This file contains fixtures and configuration for integration tests.
"""


import os
import sys
from unittest.mock import MagicMock

import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


@pytest.fixture(scope="session", autouse=True)
def mock_dependencies():
    """
    Mock dependencies that might cause issues in tests.

    This fixture is automatically used for all tests in this directory.
    """
    # Create a mock for the dependency injection container
    di_mock = MagicMock()
    di_mock.container = MagicMock()
    di_mock.reset_container = MagicMock()
    sys.modules["saplings.di"] = di_mock

    # Create a mock for the agent config
    agent_config_mock = MagicMock()
    agent_config_mock.AgentConfig = MagicMock()
    sys.modules["saplings.agent_config"] = agent_config_mock

    # Create a mock for the container module
    container_mock = MagicMock()
    container_mock.SaplingsContainer = MagicMock()
    container_mock.LifecycleScope = MagicMock()
    container_mock.Scope = MagicMock()
    sys.modules["saplings.container"] = container_mock

    # Create a mock for the container_config module
    container_config_mock = MagicMock()
    container_config_mock.configure_container = MagicMock()
    container_config_mock.initialize_container = MagicMock()
    sys.modules["saplings.container_config"] = container_config_mock

    # Create a mock for the agent module
    agent_mock = MagicMock()
    agent_mock.Agent = MagicMock()
    sys.modules["saplings.agent"] = agent_mock

    # Create mocks for all the services
    services_mock = MagicMock()
    services_mock.ModelService = MagicMock()
    services_mock.MonitoringService = MagicMock()
    services_mock.MemoryManager = MagicMock()
    services_mock.RetrievalService = MagicMock()
    services_mock.ValidatorService = MagicMock()
    services_mock.SelfHealingService = MagicMock()
    services_mock.ExecutionService = MagicMock()
    services_mock.PlannerService = MagicMock()
    services_mock.ToolService = MagicMock()
    services_mock.ModalityService = MagicMock()
    services_mock.OrchestrationService = MagicMock()
    sys.modules["saplings.services"] = services_mock
    sys.modules["saplings.services.model_service"] = MagicMock()
    sys.modules["saplings.services.monitoring_service"] = MagicMock()
    sys.modules["saplings.services.memory_manager"] = MagicMock()
    sys.modules["saplings.services.retrieval_service"] = MagicMock()
    sys.modules["saplings.services.validator_service"] = MagicMock()
    sys.modules["saplings.services.self_healing_service"] = MagicMock()
    sys.modules["saplings.services.execution_service"] = MagicMock()
    sys.modules["saplings.services.planner_service"] = MagicMock()
    sys.modules["saplings.services.tool_service"] = MagicMock()
    sys.modules["saplings.services.modality_service"] = MagicMock()
    sys.modules["saplings.services.orchestration_service"] = MagicMock()

    # Return nothing as this is an autouse fixture
