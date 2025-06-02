"""
Core Feature Stability Audit - Task 4.3

This module implements comprehensive stability tests for core features
(Agent, AgentBuilder, AgentConfig) to ensure they're truly ready for @stable annotation.

Tests focus on AgentConfig stability since Agent and AgentBuilder have complex
dependency injection requirements that are tested elsewhere.

Tests cover:
- Configuration creation and validation
- Concurrent access
- Memory pressure
- Validation performance
- API contract validation
- Error handling
"""

from __future__ import annotations

import concurrent.futures
import gc
import time

import pytest

from saplings.api.agent import AgentConfig


class TestAgentConfigStabilityScenarios:
    """Test AgentConfig under various stress conditions."""

    def setup_method(self):
        """Set up test environment."""

    def teardown_method(self):
        """Clean up test environment."""
        gc.collect()

    def test_config_creation_stability(self):
        """Test AgentConfig creation with various valid configurations."""
        # Test basic config creation
        config = AgentConfig(provider="test", model_name="test-model")

        # Config creation should succeed
        assert config is not None
        assert config.provider == "test"
        assert config.model_name == "test-model"

    def test_config_invalid_configurations(self):
        """Test AgentConfig behavior with various invalid configurations."""
        # Test invalid provider
        try:
            config = AgentConfig(provider="invalid_provider", model_name="test-model")
            # Should fail during creation due to validation
            assert False, "Should have failed with invalid provider"
        except ValueError as e:
            assert "Unsupported provider" in str(e)

        # Test invalid model name with validation
        config = AgentConfig(provider="openai", model_name="invalid-model")
        validation_result = config.validate()
        assert not validation_result.is_valid
        assert len(validation_result.suggestions) > 0

    def test_config_concurrent_access(self):
        """Test AgentConfig under concurrent access conditions."""

        # Test concurrent config creation
        def create_config():
            """Create config in thread."""
            try:
                config = AgentConfig(provider="test", model_name="test-model")
                return config is not None
            except Exception:
                return False

        # Test concurrent config creation
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_config) for _ in range(10)]
            results = [future.result(timeout=5) for future in futures]

        # All config creations should succeed
        assert all(results), "All concurrent config creations should succeed"

    def test_config_memory_pressure(self):
        """Test AgentConfig behavior under memory pressure conditions."""
        # Create multiple configs to simulate memory pressure
        configs = []
        try:
            for i in range(100):  # Create many configs
                config = AgentConfig(provider="test", model_name=f"test-model-{i}")
                configs.append(config)

                # Verify each config is properly created
                assert config is not None
                assert config.provider == "test"
                assert config.model_name == f"test-model-{i}"

        except MemoryError:
            # Expected under extreme memory pressure
            pass
        except Exception as e:
            pytest.fail(f"Unexpected error under memory pressure: {e}")
        finally:
            # Clean up
            del configs
            gc.collect()

    def test_config_validation_performance(self):
        """Test AgentConfig validation performance."""
        # Test config validation with simple scenarios
        config = AgentConfig(provider="test", model_name="test-model")

        # Validation should complete quickly
        start_time = time.time()
        result = config.validate()
        end_time = time.time()

        # Validation should be fast (< 1 second)
        assert end_time - start_time < 1.0, "Validation should be fast"

        # Result should have expected structure
        assert hasattr(result, "is_valid")
        assert hasattr(result, "message")
        assert hasattr(result, "suggestions")

    def test_config_api_contract(self):
        """Test that AgentConfig maintains stable API contract."""
        config = AgentConfig(provider="test", model_name="test-model")

        # Test required methods exist
        assert hasattr(config, "validate")

        # Test validate method contract
        result = config.validate()
        assert hasattr(result, "is_valid")
        assert hasattr(result, "message")
        assert hasattr(result, "suggestions")
        assert hasattr(result, "help_url")

    def test_config_parameter_compatibility(self):
        """Test that configuration parameters maintain compatibility."""
        config = AgentConfig(provider="test", model_name="test-model")

        # Test that all expected attributes exist
        expected_attrs = [
            "provider",
            "model_name",
            "enable_gasa",
            "enable_monitoring",
            "memory_path",
            "output_dir",
        ]

        for attr in expected_attrs:
            assert hasattr(config, attr), f"Missing expected attribute: {attr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
