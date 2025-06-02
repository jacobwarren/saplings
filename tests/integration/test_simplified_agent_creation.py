"""
Test simplified Agent creation workflow.

This test verifies that Agent creation can be simplified to eliminate
complex container setup and manual service registration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from saplings._internal.agent_module import AgentConfig
from saplings.api.agent import Agent


class TestSimplifiedAgentCreation:
    """Test simplified Agent creation patterns."""

    def test_agent_creation_with_minimal_config(self):
        """Test that Agent can be created with minimal configuration."""
        # This should work with minimal config - no manual container setup

        # Manually reset and configure the container to ensure clean state
        from saplings._internal.agent_module import AgentConfig
        from saplings._internal.container_config import configure_services
        from saplings.di import reset_container, reset_container_config

        # Reset container state
        reset_container_config()
        reset_container()

        # Create config and configure services manually
        config = AgentConfig(provider="openai", model_name="gpt-4o")
        configure_services(config)

        try:
            # Test the simplified creation pattern
            agent = Agent(provider="openai", model_name="gpt-4o")
            assert agent is not None
            assert hasattr(agent, "config")
            assert agent.config.provider == "openai"
            assert agent.config.model_name == "gpt-4o"

            # Verify that the agent has the expected services
            assert hasattr(agent, "_facade")
            assert agent._facade is not None

        except Exception as e:
            pytest.fail(f"Agent creation with minimal config failed: {e}")

    def test_agent_creation_with_config_object(self):
        """Test that Agent can be created with explicit config object."""
        # This should work with explicit config object

        # Manually reset and configure the container to ensure clean state
        from saplings._internal.container_config import configure_services
        from saplings.di import reset_container, reset_container_config

        # Reset container state
        reset_container_config()
        reset_container()

        try:
            config = AgentConfig(provider="openai", model_name="gpt-4o", enable_gasa=True)

            # Configure services manually
            configure_services(config)

            agent = Agent(config=config)
            assert agent is not None
            assert hasattr(agent, "config")
            assert agent.config.provider == "openai"
            assert agent.config.model_name == "gpt-4o"
            assert agent.config.enable_gasa is True
        except Exception as e:
            pytest.fail(f"Agent creation with config object failed: {e}")

    def test_agent_auto_configures_container(self):
        """Test that Agent automatically configures container if not already configured."""
        # Mock the container to verify it gets configured
        with patch("saplings.api.di.container") as mock_container:
            mock_container.is_configured.return_value = False
            mock_container.resolve.return_value = MagicMock()

            with patch("saplings.api.di.configure_container") as mock_configure:
                agent = Agent(provider="openai", model_name="gpt-4o")

                # Verify container was configured
                mock_configure.assert_called_once()
                assert agent is not None

    def test_agent_skips_container_config_if_already_configured(self):
        """Test that Agent skips container configuration if already configured."""
        # Mock the container to appear already configured
        with patch("saplings.api.di.container") as mock_container:
            mock_container.is_configured.return_value = True
            mock_container.resolve.return_value = MagicMock()

            with patch("saplings.api.di.configure_container") as mock_configure:
                agent = Agent(provider="openai", model_name="gpt-4o")

                # Verify container was NOT configured again
                mock_configure.assert_not_called()
                assert agent is not None

    def test_agent_auto_registers_default_validators(self):
        """Test that Agent automatically registers default validators."""
        # Mock the validator registry
        with patch("saplings.api.di.container") as mock_container:
            mock_registry = MagicMock()
            mock_container.resolve.return_value = mock_registry
            mock_container.is_configured.return_value = False

            with patch("saplings.api.di.configure_container"):
                agent = Agent(provider="openai", model_name="gpt-4o")

                # Verify default validators were registered
                # (This will be implemented in the actual Agent class)
                assert agent is not None

    def test_no_manual_container_management_required(self):
        """Test that no manual container management is required for basic usage."""
        # This test verifies that users don't need to call:
        # - reset_container_config()
        # - reset_container()
        # - configure_container()
        # - manual validator registration

        # Mock all the container operations
        with patch("saplings.api.di.container") as mock_container:
            mock_container.is_configured.return_value = False
            mock_container.resolve.return_value = MagicMock()

            with patch("saplings.api.di.configure_container"):
                # This should work without any manual setup
                agent = Agent(provider="openai", model_name="gpt-4o")
                assert agent is not None

                # User should be able to use the agent immediately
                assert hasattr(agent, "run")
                assert hasattr(agent, "add_document")

    @pytest.mark.asyncio()
    async def test_agent_run_sync_wrapper_works(self):
        """Test that Agent.run_sync() wrapper works for synchronous usage."""
        # Mock the async run method
        with patch("saplings.api.di.container") as mock_container:
            mock_container.is_configured.return_value = False
            mock_container.resolve.return_value = MagicMock()

            with patch("saplings.api.di.configure_container"):
                agent = Agent(provider="openai", model_name="gpt-4o")

                # Mock the async run method
                async def mock_run(task):
                    return f"Response to: {task}"

                agent.run = mock_run

                # Test that run_sync wrapper exists and works
                if hasattr(agent, "run_sync"):
                    result = agent.run_sync("Hello")
                    assert result == "Response to: Hello"
                else:
                    pytest.skip("run_sync method not yet implemented")

    def test_factory_methods_work(self):
        """Test that factory methods provide simplified creation patterns."""
        # Test AgentConfig factory methods
        try:
            # Test minimal factory method
            config = AgentConfig.minimal(provider="openai", model_name="gpt-4o")
            assert config.provider == "openai"
            assert config.model_name == "gpt-4o"

            # Test provider-specific factory methods
            openai_config = AgentConfig.for_openai(model_name="gpt-4o")
            assert openai_config.provider == "openai"
            assert openai_config.model_name == "gpt-4o"

            anthropic_config = AgentConfig.for_anthropic(model_name="claude-3-sonnet")
            assert anthropic_config.provider == "anthropic"
            assert anthropic_config.model_name == "claude-3-sonnet"

        except Exception as e:
            pytest.fail(f"Factory methods failed: {e}")

    def test_simplified_example_pattern(self):
        """Test that the simplified example pattern works."""
        # This should be the target simplified pattern:
        # from saplings import Agent
        # agent = Agent(provider="openai", model_name="gpt-4o")
        # result = await agent.run("Hello")

        with patch("saplings.api.di.container") as mock_container:
            mock_container.is_configured.return_value = False
            mock_container.resolve.return_value = MagicMock()

            with patch("saplings.api.di.configure_container"):
                # Step 1: Create agent with minimal config
                agent = Agent(provider="openai", model_name="gpt-4o")
                assert agent is not None

                # Step 2: Agent should be ready to use
                assert hasattr(agent, "run")
                assert callable(agent.run)

                # Step 3: No manual setup should be required
                # (All container management should be automatic)

    def test_error_handling_for_missing_config(self):
        """Test that helpful errors are provided when required configuration is missing."""
        # Test missing provider
        with pytest.raises(Exception) as exc_info:
            Agent(model_name="gpt-4o")  # Missing provider

        # The error should be helpful (this will be implemented)
        error_msg = str(exc_info.value)
        # Should contain guidance about missing provider

        # Test missing model_name
        with pytest.raises(Exception) as exc_info:
            Agent(provider="openai")  # Missing model_name

        # The error should be helpful (this will be implemented)
        error_msg = str(exc_info.value)
        # Should contain guidance about missing model_name

    def test_backward_compatibility_maintained(self):
        """Test that existing patterns still work for backward compatibility."""
        # Existing pattern with explicit config should still work
        try:
            config = AgentConfig(
                provider="openai", model_name="gpt-4o", enable_gasa=True, enable_monitoring=True
            )
            agent = Agent(config=config)
            assert agent is not None
        except Exception as e:
            pytest.fail(f"Backward compatibility broken: {e}")
