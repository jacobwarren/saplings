"""
Test for Task 10.8: Implement smart defaults for common configuration scenarios.

This test verifies that AgentConfig provides smart defaults and factory methods
for common configuration scenarios.
"""

from __future__ import annotations

import pytest

from saplings import AgentConfig


class TestTask108SmartDefaults:
    """Test Task 10.8: Implement smart defaults for common configuration scenarios."""

    def test_minimal_configuration_works(self):
        """Test that minimal configuration works with just provider and model."""
        # Should work with just provider and model_name
        config = AgentConfig(provider="openai", model_name="gpt-4o")

        assert config.provider == "openai"
        assert config.model_name == "gpt-4o"

        # Should have sensible defaults
        assert config.memory_path == "./agent_memory"
        assert config.output_dir == "./agent_output"
        assert isinstance(config.enable_gasa, bool)
        assert isinstance(config.enable_monitoring, bool)

        print("✅ Minimal configuration works with sensible defaults")

    def test_factory_method_minimal(self):
        """Test AgentConfig.minimal() factory method."""
        config = AgentConfig.minimal("openai", "gpt-4o")

        assert config.provider == "openai"
        assert config.model_name == "gpt-4o"

        # Minimal should disable advanced features
        assert config.enable_gasa == False
        assert config.enable_monitoring == False
        assert config.enable_self_healing == False
        assert config.enable_tool_factory == False

        print("✅ AgentConfig.minimal() factory method works")

    def test_factory_method_standard(self):
        """Test AgentConfig.standard() factory method."""
        config = AgentConfig.standard("openai", "gpt-4o")

        assert config.provider == "openai"
        assert config.model_name == "gpt-4o"

        # Standard should enable some features but not all
        assert config.enable_gasa == True
        assert config.enable_monitoring == True
        assert config.enable_self_healing == False  # Not enabled in standard
        assert config.enable_tool_factory == True

        print("✅ AgentConfig.standard() factory method works")

    def test_factory_method_full_featured(self):
        """Test AgentConfig.full_featured() factory method."""
        config = AgentConfig.full_featured("openai", "gpt-4o")

        assert config.provider == "openai"
        assert config.model_name == "gpt-4o"

        # Full-featured should enable all features
        assert config.enable_gasa == True
        assert config.enable_monitoring == True
        assert config.enable_self_healing == True
        assert config.enable_tool_factory == True

        print("✅ AgentConfig.full_featured() factory method works")

    def test_provider_specific_factory_openai(self):
        """Test AgentConfig.for_openai() factory method."""
        config = AgentConfig.for_openai("gpt-4o")

        assert config.provider == "openai"
        assert config.model_name == "gpt-4o"

        # Should have OpenAI-specific optimizations
        assert config.enable_gasa == True

        print("✅ AgentConfig.for_openai() factory method works")

    def test_provider_specific_factory_anthropic(self):
        """Test AgentConfig.for_anthropic() factory method."""
        config = AgentConfig.for_anthropic("claude-3-opus")

        assert config.provider == "anthropic"
        assert config.model_name == "claude-3-opus"

        # Should have Anthropic-specific optimizations
        assert config.enable_gasa == True

        print("✅ AgentConfig.for_anthropic() factory method works")

    def test_provider_specific_factory_vllm(self):
        """Test AgentConfig.for_vllm() factory method."""
        config = AgentConfig.for_vllm("Qwen/Qwen3-7B-Instruct")

        assert config.provider == "vllm"
        assert config.model_name == "Qwen/Qwen3-7B-Instruct"

        # Should have vLLM-specific optimizations
        assert config.enable_gasa == True

        print("✅ AgentConfig.for_vllm() factory method works")

    def test_factory_methods_accept_overrides(self):
        """Test that factory methods accept parameter overrides."""
        # Test minimal with overrides
        config = AgentConfig.minimal("openai", "gpt-4o", enable_monitoring=True)
        assert config.enable_monitoring == True  # Override should work
        assert config.enable_gasa == False  # Other defaults should remain

        # Test for_openai with overrides
        config = AgentConfig.for_openai("gpt-4o", memory_path="./custom_memory")
        assert config.memory_path == "./custom_memory"  # Override should work
        assert config.provider == "openai"  # Provider should remain

        print("✅ Factory methods accept parameter overrides")

    def test_configuration_presets_exist(self):
        """Test that configuration presets are defined."""
        # Check that presets exist as class variables
        assert hasattr(AgentConfig, "PRESET_MINIMAL")
        assert hasattr(AgentConfig, "PRESET_STANDARD")
        assert hasattr(AgentConfig, "PRESET_FULL_FEATURED")
        assert hasattr(AgentConfig, "PRESET_OPENAI")

        # Check that presets are dictionaries
        assert isinstance(AgentConfig.PRESET_MINIMAL, dict)
        assert isinstance(AgentConfig.PRESET_STANDARD, dict)
        assert isinstance(AgentConfig.PRESET_FULL_FEATURED, dict)
        assert isinstance(AgentConfig.PRESET_OPENAI, dict)

        print("✅ Configuration presets are defined")

    def test_smart_defaults_progression(self):
        """Test that defaults progress logically from minimal to full-featured."""
        minimal = AgentConfig.minimal("openai", "gpt-4o")
        standard = AgentConfig.standard("openai", "gpt-4o")
        full = AgentConfig.full_featured("openai", "gpt-4o")

        # Minimal should have fewest features enabled
        minimal_features = sum(
            [
                minimal.enable_gasa,
                minimal.enable_monitoring,
                minimal.enable_self_healing,
                minimal.enable_tool_factory,
            ]
        )

        # Standard should have more features than minimal
        standard_features = sum(
            [
                standard.enable_gasa,
                standard.enable_monitoring,
                standard.enable_self_healing,
                standard.enable_tool_factory,
            ]
        )

        # Full should have most features enabled
        full_features = sum(
            [
                full.enable_gasa,
                full.enable_monitoring,
                full.enable_self_healing,
                full.enable_tool_factory,
            ]
        )

        assert minimal_features <= standard_features <= full_features
        print(
            f"Feature progression: minimal({minimal_features}) ≤ standard({standard_features}) ≤ full({full_features})"
        )
        print("✅ Smart defaults progress logically")

    def test_error_messages_are_helpful(self):
        """Test that error messages provide helpful guidance."""
        # Test missing provider
        with pytest.raises(ValueError) as exc_info:
            AgentConfig(provider="", model_name="gpt-4o")

        error_msg = str(exc_info.value)
        assert "provider" in error_msg.lower()
        assert "example" in error_msg.lower()  # Should provide examples

        # Test missing model_name
        with pytest.raises(ValueError) as exc_info:
            AgentConfig(provider="openai", model_name="")

        error_msg = str(exc_info.value)
        assert "model_name" in error_msg.lower()
        assert "example" in error_msg.lower()  # Should provide examples

        # Test unsupported provider
        with pytest.raises(ValueError) as exc_info:
            AgentConfig(provider="unsupported", model_name="some-model")

        error_msg = str(exc_info.value)
        assert "unsupported" in error_msg.lower()
        assert "supported providers" in error_msg.lower()

        print("✅ Error messages are helpful and provide guidance")

    def test_configuration_explanation_and_comparison(self):
        """Test that configuration explanation and comparison methods work."""
        # Test explain method
        config = AgentConfig(provider="openai", model_name="gpt-4o")
        explanation = config.explain()

        assert "openai" in explanation.lower()
        assert "gpt-4o" in explanation.lower()
        assert "gasa" in explanation.lower()
        assert "monitoring" in explanation.lower()

        print("✅ Configuration explanation works")

        # Test compare method with identical configs
        config1 = AgentConfig.minimal("openai", "gpt-4o")
        config2 = AgentConfig.minimal("openai", "gpt-4o")
        comparison = config1.compare(config2)

        assert "identical" in comparison.lower()
        print("✅ Configuration comparison works for identical configs")

        # Test compare method with different configs
        config3 = AgentConfig.standard("anthropic", "claude-3-opus")
        comparison = config1.compare(config3)

        assert "differences" in comparison.lower()
        assert "provider" in comparison.lower()
        assert "model" in comparison.lower()

        print("✅ Configuration comparison works for different configs")

        # Test compare with non-AgentConfig object
        comparison = config1.compare("not a config")
        assert "cannot compare" in comparison.lower()

        print("✅ Configuration comparison handles invalid input")

    def test_configuration_summary(self):
        """Test that we can summarize the smart defaults functionality."""
        print("\n=== Smart Defaults Summary ===")

        # Test all factory methods
        factory_methods = [
            ("minimal", AgentConfig.minimal),
            ("standard", AgentConfig.standard),
            ("full_featured", AgentConfig.full_featured),
            ("for_openai", AgentConfig.for_openai),
            ("for_anthropic", AgentConfig.for_anthropic),
            ("for_vllm", AgentConfig.for_vllm),
        ]

        working_methods = []
        for name, method in factory_methods:
            try:
                if name.startswith("for_"):
                    config = method("test-model")
                else:
                    config = method("openai", "test-model")
                working_methods.append(name)
                print(f"  ✅ {name}() - {config.provider}")
            except Exception as e:
                print(f"  ❌ {name}() - Error: {e}")

        print(f"\nWorking factory methods: {len(working_methods)}/{len(factory_methods)}")

        # Test basic configuration
        try:
            config = AgentConfig(provider="openai", model_name="gpt-4o")
            print("✅ Basic configuration works")
            print(f"  - Provider: {config.provider}")
            print(f"  - Model: {config.model_name}")
            print(f"  - Memory path: {config.memory_path}")
            print(f"  - GASA enabled: {config.enable_gasa}")
        except Exception as e:
            print(f"❌ Basic configuration failed: {e}")

        assert (
            len(working_methods) >= 4
        ), f"Expected at least 4 working factory methods, got {len(working_methods)}"


if __name__ == "__main__":
    # Run the tests when script is executed directly
    pytest.main([__file__, "-v"])
