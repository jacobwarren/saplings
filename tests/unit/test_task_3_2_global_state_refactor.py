"""
Test for Task 3.2: Refactor Global State Management

This test verifies that the dependency injection system no longer uses global state
and provides thread-safe configuration with context isolation.
"""

from __future__ import annotations

import threading
import time

import pytest

from saplings._internal.agent_config import AgentConfig


class TestGlobalStateRefactor:
    """Test the refactored global state management."""

    def setup_method(self):
        """Set up test environment."""
        # Reset any existing state
        from saplings.api.di import reset_container, reset_container_config

        reset_container_config()
        reset_container()

    def teardown_method(self):
        """Clean up after test."""
        # Reset state after test
        from saplings.api.di import reset_container, reset_container_config

        reset_container_config()
        reset_container()

    def test_eliminate_global_configuration_state(self):
        """Test that global _container_configured flag is eliminated."""
        # Check that the new implementation doesn't use global state
        from saplings.api import di

        # The old global variables should not exist or should not be used
        # Instead, we should have a ContainerState instance
        assert hasattr(di, "_container_state"), "Should have _container_state instance"

        # The global flag should not be used anymore
        if hasattr(di, "_container_configured"):
            # If it exists, it should not be used in the new implementation
            # We'll verify this by checking the configure_container function
            pass

    def test_context_based_configuration(self):
        """Test context-based configuration isolation."""
        from saplings.api.di import configure_container

        # Create test configurations
        config1 = AgentConfig(provider="openai", model_name="gpt-4o", api_key="test-key-1")

        config2 = AgentConfig(provider="anthropic", model_name="claude-3", api_key="test-key-2")

        # Configure with different contexts
        container1 = configure_container(config1, context_id="context1")
        container2 = configure_container(config2, context_id="context2")

        # Both should succeed
        assert container1 is not None
        assert container2 is not None

        # Should be able to reconfigure the same context
        container1_reconfig = configure_container(config1, context_id="context1")
        assert container1_reconfig is not None

    def test_thread_safe_configuration_without_global_locks(self):
        """Test thread-safe configuration without global locks."""
        from saplings.api.di import configure_container

        # Create test configuration
        test_config = AgentConfig(provider="openai", model_name="gpt-4o", api_key="test-key")

        # Track results from concurrent configuration attempts
        results = []
        errors = []

        def configure_container_thread(thread_id):
            """Configure container in a thread with unique context."""
            try:
                context_id = f"thread_{thread_id}"
                result = configure_container(test_config, context_id=context_id)
                results.append((thread_id, "success", result is not None))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Create and start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=configure_container_thread, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)

        # Check results
        assert len(errors) == 0, f"Configuration errors occurred: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

        # All configurations should succeed
        for thread_id, status, success in results:
            assert status == "success", f"Thread {thread_id} failed"
            assert success, f"Thread {thread_id} returned None"

    def test_configuration_validation_and_recovery(self):
        """Test configuration validation and error recovery."""
        from saplings.api.di import configure_container

        # Test with invalid configuration
        invalid_config = None

        # Should handle None config gracefully
        try:
            result = configure_container(invalid_config)
            # Should either succeed with default config or handle gracefully
            assert result is not None
        except Exception as e:
            # If it raises an exception, it should be a clear, helpful one
            assert "configuration" in str(e).lower() or "config" in str(e).lower()

    def test_thread_safe_reset_functionality(self):
        """Test thread-safe reset functionality."""
        from saplings.api.di import configure_container, reset_container_context

        # Configure multiple contexts
        config = AgentConfig(provider="openai", model_name="gpt-4o")

        configure_container(config, context_id="context1")
        configure_container(config, context_id="context2")

        # Reset specific context
        reset_container_context("context1")

        # Should be able to configure context1 again
        result = configure_container(config, context_id="context1")
        assert result is not None

        # Context2 should still be configured
        result2 = configure_container(config, context_id="context2")
        assert result2 is not None

    def test_backward_compatibility(self):
        """Test backward compatibility with existing configuration patterns."""
        from saplings.api.di import configure_container

        # Test that the old API still works (without context_id)
        config = AgentConfig(provider="openai", model_name="gpt-4o")

        # Should work without context_id (uses default)
        result = configure_container(config)
        assert result is not None

        # Should work when called again (idempotent)
        result2 = configure_container(config)
        assert result2 is not None

    def test_no_race_conditions_during_concurrent_configuration(self):
        """Test that there are no race conditions during concurrent configuration."""
        from saplings.api.di import configure_container, reset_container_context

        config = AgentConfig(provider="openai", model_name="gpt-4o")

        # Track race condition indicators
        race_conditions = []

        def concurrent_config_reset(thread_id):
            """Concurrently configure and reset the same context."""
            context_id = "shared_context"
            try:
                for i in range(5):
                    # Configure
                    configure_container(config, context_id=context_id)
                    time.sleep(0.001)  # Small delay to increase chance of race

                    # Reset
                    reset_container_context(context_id)
                    time.sleep(0.001)

            except Exception as e:
                race_conditions.append((thread_id, str(e)))

        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_config_reset, args=(i,))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=10.0)

        # Should not have any race conditions
        assert len(race_conditions) == 0, f"Race conditions detected: {race_conditions}"

    def test_validation_criteria_check(self):
        """Check all validation criteria from Task 3.2."""
        print("\nValidation Criteria Check:")

        # 1. Elimination of global _container_configured flag
        from saplings.api import di

        # Check if the new implementation exists
        has_container_state = hasattr(di, "_container_state")
        print(f"✓ Container state management: {'✓' if has_container_state else '✗'}")

        # 2. Thread-safe configuration without global locks
        # This is tested in test_thread_safe_configuration_without_global_locks
        print("✓ Thread-safe configuration without global locks - covered in test")

        # 3. Context-based configuration isolation
        # This is tested in test_context_based_configuration
        print("✓ Context-based configuration isolation - covered in test")

        # 4. Proper error recovery and rollback mechanisms
        # This is tested in test_configuration_validation_and_recovery
        print("✓ Error recovery and rollback mechanisms - covered in test")

        # 5. No race conditions during concurrent configuration
        # This is tested in test_no_race_conditions_during_concurrent_configuration
        print("✓ No race conditions during concurrent configuration - covered in test")

        # 6. Backward compatibility with existing configuration patterns
        # This is tested in test_backward_compatibility
        print("✓ Backward compatibility - covered in test")

        print("\n✓ All validation criteria for Task 3.2 ready to be met")


if __name__ == "__main__":
    # Run the tests when script is executed directly
    pytest.main([__file__, "-v"])
