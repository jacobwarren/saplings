"""
Test for Fallback Mechanisms Implementation

This test validates the fallback mechanisms for optional services as specified in Task 3.4.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from saplings._internal.fallback_services import (
    NullGASAService,
    NullMonitoringService,
    NullOrchestrationService,
    NullSelfHealingService,
)
from saplings._internal.service_availability import (
    ServiceAvailability,
)
from saplings._internal.service_registry import (
    configure_services_with_fallbacks,
)


class TestServiceAvailability:
    """Test service availability detection and conditional registration."""

    def setup_method(self):
        """Set up test environment."""
        # Reset any global state
        global SERVICE_REGISTRY
        SERVICE_REGISTRY.clear()

    def test_service_availability_initialization(self):
        """Test that ServiceAvailability initializes correctly."""
        availability = ServiceAvailability()

        assert isinstance(availability.available_services, dict)
        assert isinstance(availability.fallback_services, dict)
        assert len(availability.available_services) == 0
        assert len(availability.fallback_services) == 0

    def test_register_conditional_service_available(self):
        """Test registering service when condition is met."""
        availability = ServiceAvailability()

        # Mock implementation that's available
        mock_implementation = MagicMock()
        condition = lambda: True  # Always available

        availability.register_conditional_service("IGASAService", mock_implementation, condition)

        assert "IGASAService" in availability.available_services
        assert availability.available_services["IGASAService"] == mock_implementation

    def test_register_conditional_service_unavailable(self):
        """Test registering service when condition is not met."""
        availability = ServiceAvailability()

        # Set up fallback
        fallback_service = MagicMock()
        availability.fallback_services["IGASAService"] = fallback_service

        # Mock implementation that's not available
        mock_implementation = MagicMock()
        condition = lambda: False  # Not available

        availability.register_conditional_service("IGASAService", mock_implementation, condition)

        assert "IGASAService" in availability.available_services
        assert availability.available_services["IGASAService"] == fallback_service

    def test_register_conditional_service_no_fallback(self):
        """Test registering service when condition not met and no fallback."""
        availability = ServiceAvailability()

        # Mock implementation that's not available
        mock_implementation = MagicMock()
        condition = lambda: False  # Not available

        availability.register_conditional_service("IGASAService", mock_implementation, condition)

        assert "IGASAService" in availability.available_services
        assert availability.available_services["IGASAService"] is None


class TestFallbackServices:
    """Test fallback service implementations."""

    def test_null_gasa_service(self):
        """Test NullGASAService fallback implementation."""
        service = NullGASAService()

        # Test create_mask method
        result = service.create_mask(graph={"nodes": []}, tokens=["test", "tokens"])
        assert result is None  # Should return None for no mask

    def test_null_monitoring_service(self):
        """Test NullMonitoringService fallback implementation."""
        service = NullMonitoringService()

        # Test log_event method
        event = MagicMock()
        event.event_type = "test_event"

        # Should not raise an exception
        service.log_event(event)

    def test_null_self_healing_service(self):
        """Test NullSelfHealingService fallback implementation."""
        service = NullSelfHealingService()

        # Test fix_error method
        error = Exception("test error")
        context = {"task_id": "test"}

        result = service.fix_error(error, context)
        assert result is not None
        assert "fixed" in result
        assert result["fixed"] is False  # Null service doesn't actually fix

    def test_null_orchestration_service(self):
        """Test NullOrchestrationService fallback implementation."""
        service = NullOrchestrationService()

        # Test run_graph method
        graph = {"nodes": [], "edges": []}
        inputs = {"input1": "value1"}

        result = service.run_graph(graph, inputs)
        assert result is not None
        assert "status" in result
        assert result["status"] == "skipped"  # Null service skips execution


class TestGracefulDegradation:
    """Test graceful degradation when optional services unavailable."""

    def setup_method(self):
        """Set up test environment."""
        # Reset any global state
        global SERVICE_REGISTRY
        SERVICE_REGISTRY.clear()

    @patch("saplings._internal.optional_deps.check_feature_availability")
    def test_configure_services_with_gasa_available(self, mock_check_features):
        """Test service configuration when GASA is available."""
        # Mock GASA as available
        mock_check_features.return_value = {"gasa": True}

        config = MagicMock()

        # This should use real GASA service (or mock of real service)
        with patch("saplings._internal.service_registry.create_gasa_service") as mock_create:
            mock_gasa = MagicMock()
            mock_create.return_value = mock_gasa

            configure_services_with_fallbacks(config)

            mock_create.assert_called_once_with(config)

    @patch("saplings._internal.optional_deps.check_feature_availability")
    def test_configure_services_with_gasa_unavailable(self, mock_check_features):
        """Test service configuration when GASA is unavailable."""
        # Mock GASA as unavailable
        mock_check_features.return_value = {"gasa": False}

        config = MagicMock()

        # This should use fallback GASA service
        with patch("saplings._internal.service_registry.create_gasa_service") as mock_create:
            mock_create.side_effect = ImportError("GASA dependencies not available")

            # Should not raise an exception
            configure_services_with_fallbacks(config)

    def test_fallback_services_implement_interfaces(self):
        """Test that fallback services properly implement expected interfaces."""
        # Test that all fallback services have the required methods

        # GASA Service
        gasa_service = NullGASAService()
        assert hasattr(gasa_service, "create_mask")
        assert callable(gasa_service.create_mask)

        # Monitoring Service
        monitoring_service = NullMonitoringService()
        assert hasattr(monitoring_service, "log_event")
        assert callable(monitoring_service.log_event)

        # Self-Healing Service
        self_healing_service = NullSelfHealingService()
        assert hasattr(self_healing_service, "fix_error")
        assert callable(self_healing_service.fix_error)

        # Orchestration Service
        orchestration_service = NullOrchestrationService()
        assert hasattr(orchestration_service, "run_graph")
        assert callable(orchestration_service.run_graph)

    def test_clear_warnings_about_missing_functionality(self):
        """Test that clear warnings are provided about missing functionality."""
        with patch("saplings._internal.optional_deps.check_feature_availability") as mock_check:
            mock_check.return_value = {"gasa": False, "monitoring": False}

            config = MagicMock()

            # Should log warnings about missing functionality
            with patch("saplings._internal.service_registry.logger") as mock_logger:
                configure_services_with_fallbacks(config)

                # Should have logged warnings about fallbacks
                assert mock_logger.warning.called

                # Check that specific warning messages were logged
                warning_calls = mock_logger.warning.call_args_list
                warning_messages = [call[0][0] for call in warning_calls]

                assert any("GASA dependencies not available" in msg for msg in warning_messages)
                assert any(
                    "Monitoring dependencies not available" in msg for msg in warning_messages
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
