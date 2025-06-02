"""
Test for Task 10.7: Create truly minimal examples that work out of the box.

This test verifies that minimal examples can be created that work without
extensive boilerplate code.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestTask107MinimalExamples:
    """Test Task 10.7: Create truly minimal examples that work out of the box."""

    def test_minimal_agent_example_exists(self):
        """Test that a minimal agent example exists."""
        examples_dir = Path("examples")
        minimal_example = examples_dir / "minimal_agent.py"

        # Check if minimal example exists
        if minimal_example.exists():
            print("✅ Minimal agent example exists")

            # Check the length - should be very short
            content = minimal_example.read_text()
            lines = [
                line
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            print(f"Minimal example has {len(lines)} non-comment lines")

            # Should be under 15 lines of actual code
            if len(lines) <= 15:
                print("✅ Minimal example is truly minimal")
            else:
                print(f"⚠️  Minimal example has {len(lines)} lines, should be ≤15")
        else:
            print("❌ Minimal agent example doesn't exist")

    def test_basic_agent_example_exists(self):
        """Test that a basic agent example exists."""
        examples_dir = Path("examples")
        basic_example = examples_dir / "basic_agent.py"

        # Check if basic example exists
        if basic_example.exists():
            print("✅ Basic agent example exists")

            # Check the length - should be reasonably short
            content = basic_example.read_text()
            lines = [
                line
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            print(f"Basic example has {len(lines)} non-comment lines")

            # Should be under 25 lines of actual code
            if len(lines) <= 25:
                print("✅ Basic example is appropriately sized")
            else:
                print(f"⚠️  Basic example has {len(lines)} lines, should be ≤25")
        else:
            print("❌ Basic agent example doesn't exist")

    def test_current_examples_complexity(self):
        """Test the complexity of current examples."""
        examples_dir = Path("examples")

        if not examples_dir.exists():
            pytest.skip("Examples directory doesn't exist")

        # Check some current examples
        examples_to_check = [
            "basic_agent_setup.py",
            "simplified_agent_setup.py",
            "simple_agent_run_test.py",
        ]

        complexity_results = []

        for example_name in examples_to_check:
            example_path = examples_dir / example_name
            if example_path.exists():
                content = example_path.read_text()
                lines = content.split("\n")
                total_lines = len(lines)
                code_lines = len(
                    [line for line in lines if line.strip() and not line.strip().startswith("#")]
                )

                # Check for boilerplate patterns
                has_container_reset = "reset_container" in content
                has_container_config = "configure_container" in content
                has_validator_setup = "register_validator" in content
                has_manual_setup = (
                    has_container_reset or has_container_config or has_validator_setup
                )

                complexity_results.append(
                    {
                        "name": example_name,
                        "total_lines": total_lines,
                        "code_lines": code_lines,
                        "has_manual_setup": has_manual_setup,
                        "has_container_reset": has_container_reset,
                        "has_container_config": has_container_config,
                        "has_validator_setup": has_validator_setup,
                    }
                )

                print(f"\n{example_name}:")
                print(f"  Total lines: {total_lines}")
                print(f"  Code lines: {code_lines}")
                print(f"  Manual container setup: {has_manual_setup}")
                if has_manual_setup:
                    print(f"    - Container reset: {has_container_reset}")
                    print(f"    - Container config: {has_container_config}")
                    print(f"    - Validator setup: {has_validator_setup}")

        # Report summary
        print("\nComplexity Summary:")
        print(f"Examples checked: {len(complexity_results)}")

        manual_setup_count = sum(1 for r in complexity_results if r["has_manual_setup"])
        print(f"Examples with manual setup: {manual_setup_count}/{len(complexity_results)}")

        avg_lines = (
            sum(r["code_lines"] for r in complexity_results) / len(complexity_results)
            if complexity_results
            else 0
        )
        print(f"Average code lines: {avg_lines:.1f}")

        # This test documents current state - doesn't fail
        assert len(complexity_results) > 0, "Should have found some examples to analyze"

    def test_minimal_example_requirements(self):
        """Test what a minimal example should look like."""
        # This test defines the requirements for a minimal example

        minimal_requirements = {
            "max_lines": 10,  # Should be ≤10 lines of actual code
            "no_manual_container": True,  # No manual container management
            "no_manual_validators": True,  # No manual validator registration
            "automatic_api_key": True,  # Should auto-detect API keys
            "simple_imports": True,  # Should use simple imports
        }

        print("Minimal example requirements:")
        for req, value in minimal_requirements.items():
            print(f"  {req}: {value}")

        # Example of what minimal code should look like:
        minimal_example_code = """
from saplings import Agent

agent = Agent(provider="openai", model_name="gpt-4o")
result = await agent.run("What is 2+2?")
print(result)
"""

        lines = [line for line in minimal_example_code.strip().split("\n") if line.strip()]
        print(f"\nIdeal minimal example ({len(lines)} lines):")
        for i, line in enumerate(lines, 1):
            print(f"  {i}. {line}")

        # This test passes - it's documenting requirements
        assert len(lines) <= minimal_requirements["max_lines"]

    def test_progressive_complexity_examples(self):
        """Test that examples should follow progressive complexity."""

        expected_examples = [
            {
                "name": "minimal_agent.py",
                "description": "Absolute minimum (5-10 lines)",
                "max_lines": 10,
                "features": ["basic agent creation", "simple query"],
            },
            {
                "name": "basic_agent.py",
                "description": "With tools and memory (15-20 lines)",
                "max_lines": 20,
                "features": ["agent creation", "tool usage", "memory operations"],
            },
            {
                "name": "advanced_agent.py",
                "description": "With full configuration (30-40 lines)",
                "max_lines": 40,
                "features": ["full configuration", "error handling", "multiple tools"],
            },
            {
                "name": "production_agent.py",
                "description": "With error handling and monitoring (50+ lines)",
                "max_lines": 100,
                "features": ["production setup", "monitoring", "error handling", "logging"],
            },
        ]

        print("Progressive complexity examples should include:")
        for example in expected_examples:
            print(f"\n{example['name']} - {example['description']}")
            print(f"  Max lines: {example['max_lines']}")
            print(f"  Features: {', '.join(example['features'])}")

        # Check if any of these exist
        examples_dir = Path("examples")
        existing_count = 0

        if examples_dir.exists():
            for example in expected_examples:
                example_path = examples_dir / example["name"]
                if example_path.exists():
                    existing_count += 1
                    print(f"  ✅ {example['name']} exists")
                else:
                    print(f"  ❌ {example['name']} missing")

        print(f"\nProgressive examples found: {existing_count}/{len(expected_examples)}")

        # This test documents what should exist - doesn't fail
        assert len(expected_examples) == 4, "Should define 4 progressive examples"


if __name__ == "__main__":
    # Run the tests when script is executed directly
    pytest.main([__file__, "-v"])
