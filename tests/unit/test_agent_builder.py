"""
Unit tests for the AgentBuilder class.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from saplings.agent_builder import AgentBuilder


class TestAgentBuilder:
    """Tests for the AgentBuilder class."""

    def test_should_use_container_with_gasa(self):
        """Test that _should_use_container returns True when GASA is enabled."""
        builder = AgentBuilder()
        builder.with_gasa_enabled(True)
        assert builder._should_use_container() is True

    def test_should_use_container_with_monitoring(self):
        """Test that _should_use_container returns True when monitoring is enabled."""
        builder = AgentBuilder()
        builder.with_monitoring_enabled(True)
        assert builder._should_use_container() is True

    def test_should_use_container_with_self_healing(self):
        """Test that _should_use_container returns True when self-healing is enabled."""
        builder = AgentBuilder()
        builder.with_config({"enable_self_healing": True})
        assert builder._should_use_container() is True

    def test_should_not_use_container_with_basic_features(self):
        """Test that _should_use_container returns False with basic features."""
        builder = AgentBuilder()
        builder.with_provider("openai")
        builder.with_model_name("gpt-4o")
        assert builder._should_use_container() is False

    def test_build_with_auto_container_determination(self):
        """Test that build() automatically determines whether to use the container."""
        builder = AgentBuilder()
        builder.with_provider("openai")
        builder.with_model_name("gpt-4o")

        # Mock the _should_use_container method to return False
        with patch.object(builder, "_should_use_container", return_value=False):
            # Mock the _build_without_container method
            with patch.object(builder, "_build_without_container") as mock_build_without:
                # Mock the initialize_container function
                with patch("saplings.agent_builder.initialize_container"):
                    builder.build()
                    # Assert that _build_without_container was called
                    mock_build_without.assert_called_once()

    def test_build_with_auto_container_determination_advanced(self):
        """Test that build() automatically uses the container for advanced features."""
        builder = AgentBuilder()
        builder.with_provider("openai")
        builder.with_model_name("gpt-4o")
        builder.with_gasa_enabled(True)

        # Mock the _should_use_container method to return True
        with patch.object(builder, "_should_use_container", return_value=True):
            # Mock the Agent class
            with patch("saplings.agent_builder.Agent") as mock_agent:
                # Mock the initialize_container function
                with patch("saplings.agent_builder.initialize_container"):
                    builder.build()
                    # Assert that Agent was instantiated
                    mock_agent.assert_called_once()

    def test_build_with_explicit_container_true(self):
        """Test that build(use_container=True) uses the container."""
        builder = AgentBuilder()
        builder.with_provider("openai")
        builder.with_model_name("gpt-4o")

        # Mock the _should_use_container method to ensure it's not called
        with patch.object(builder, "_should_use_container") as mock_should_use:
            # Mock the Agent class
            with patch("saplings.agent_builder.Agent") as mock_agent:
                # Mock the initialize_container function
                with patch("saplings.agent_builder.initialize_container"):
                    builder.build(use_container=True)
                    # Assert that _should_use_container was not called
                    mock_should_use.assert_not_called()
                    # Assert that Agent was instantiated
                    mock_agent.assert_called_once()

    def test_build_with_explicit_container_false(self):
        """Test that build(use_container=False) doesn't use the container."""
        builder = AgentBuilder()
        builder.with_provider("openai")
        builder.with_model_name("gpt-4o")

        # Mock the _should_use_container method to ensure it's not called
        with patch.object(builder, "_should_use_container") as mock_should_use:
            # Mock the _build_without_container method
            with patch.object(builder, "_build_without_container") as mock_build_without:
                builder.build(use_container=False)
                # Assert that _should_use_container was not called
                mock_should_use.assert_not_called()
                # Assert that _build_without_container was called
                mock_build_without.assert_called_once()

    def test_build_with_deprecation_warning(self):
        """Test that build(use_container=...) issues a deprecation warning."""
        builder = AgentBuilder()
        builder.with_provider("openai")
        builder.with_model_name("gpt-4o")

        # Mock the _build_without_container method
        with patch.object(builder, "_build_without_container"):
            # Mock the initialize_container function
            with patch("saplings.agent_builder.initialize_container"):
                # Check for deprecation warning
                with pytest.warns(DeprecationWarning):
                    builder.build(use_container=False)
