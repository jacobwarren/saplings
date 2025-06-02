"""
Test for Task 10.9: Add comprehensive getting started documentation.

This test verifies that comprehensive getting started documentation exists
and covers all essential topics for new users.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestTask109Documentation:
    """Test Task 10.9: Add comprehensive getting started documentation."""

    def test_external_docs_directory_exists(self):
        """Test that external-docs directory exists."""
        external_docs_path = Path("external-docs")
        assert external_docs_path.exists(), "external-docs directory should exist"
        assert external_docs_path.is_dir(), "external-docs should be a directory"
        print("✅ external-docs directory exists")

    def test_getting_started_guide_exists(self):
        """Test that getting started guide exists and has essential content."""
        getting_started_path = Path("external-docs/getting-started.md")
        assert getting_started_path.exists(), "getting-started.md should exist"

        content = getting_started_path.read_text()

        # Check for essential sections
        essential_sections = [
            "What is Saplings",
            "Quick Start",
            "Install",
            "API key",
            "Create your first agent",
            "Next Steps",
            "Common Patterns",
            "Built-in Tools",
            "Troubleshooting",
        ]

        for section in essential_sections:
            assert (
                section.lower() in content.lower()
            ), f"Getting started guide should mention '{section}'"

        # Check for code examples
        assert "```python" in content, "Should contain Python code examples"
        assert "AgentConfig" in content, "Should show AgentConfig usage"
        assert "Agent(" in content, "Should show Agent creation"

        print("✅ Getting started guide exists with essential content")

    def test_quick_start_tutorial_exists(self):
        """Test that quick start tutorial exists and is comprehensive."""
        tutorial_path = Path("external-docs/quick-start-tutorial.md")
        assert tutorial_path.exists(), "quick-start-tutorial.md should exist"

        content = tutorial_path.read_text()

        # Check for tutorial structure
        tutorial_elements = [
            "15-Minute",
            "What We'll Build",
            "Prerequisites",
            "Step 1",
            "Step 2",
            "Step 3",
            "Installation",
            "Basic Agent",
            "Web Search",
            "Memory",
            "Tools",
            "Production-Ready",
        ]

        for element in tutorial_elements:
            assert element in content, f"Tutorial should include '{element}'"

        # Check for practical examples
        assert "DuckDuckGoSearchTool" in content, "Should include search tool example"
        assert "memory_path" in content, "Should show memory configuration"
        assert "asyncio" in content, "Should show async usage"

        print("✅ Quick start tutorial exists with comprehensive content")

    def test_troubleshooting_guide_exists(self):
        """Test that troubleshooting guide exists and covers common issues."""
        troubleshooting_path = Path("external-docs/troubleshooting.md")
        assert troubleshooting_path.exists(), "troubleshooting.md should exist"

        content = troubleshooting_path.read_text()

        # Check for common issue categories
        issue_categories = [
            "Installation Issues",
            "API Key Issues",
            "Import and Dependency Issues",
            "Agent Creation Issues",
            "Memory and Storage Issues",
            "Tool Issues",
            "Performance Issues",
            "Getting Help",
        ]

        for category in issue_categories:
            assert category in content, f"Troubleshooting should cover '{category}'"

        # Check for specific common errors
        common_errors = [
            "No module named 'saplings'",
            "API key not found",
            "Rate limit exceeded",
            "ImportError",
            "Permission denied",
            "ServiceNotRegisteredError",
        ]

        for error in common_errors:
            assert error in content, f"Should cover common error: '{error}'"

        print("✅ Troubleshooting guide exists with comprehensive coverage")

    def test_installation_guide_exists(self):
        """Test that installation guide exists and covers all platforms."""
        installation_path = Path("external-docs/installation.md")
        assert installation_path.exists(), "installation.md should exist"

        content = installation_path.read_text()

        # Check for installation methods
        installation_methods = [
            "Quick Installation",
            "Optional Dependencies",
            "Platform-Specific",
            "Windows",
            "macOS",
            "Linux",
            "Virtual Environment",
            "Development Installation",
            "Verification",
        ]

        for method in installation_methods:
            assert method in content, f"Installation guide should cover '{method}'"

        # Check for dependency groups
        dependency_groups = ["[tools]", "[viz]", "[monitoring]", "[dev]"]

        for group in dependency_groups:
            assert group in content, f"Should mention dependency group: '{group}'"

        print("✅ Installation guide exists with comprehensive coverage")

    def test_documentation_structure_is_logical(self):
        """Test that documentation follows a logical progression."""
        docs = [
            "installation.md",
            "getting-started.md",
            "quick-start-tutorial.md",
            "troubleshooting.md",
        ]

        for doc in docs:
            doc_path = Path(f"external-docs/{doc}")
            assert doc_path.exists(), f"{doc} should exist"

        # Check that getting started references other docs
        getting_started_content = Path("external-docs/getting-started.md").read_text()

        # Should reference other guides
        references = ["troubleshooting", "installation", "examples", "docs"]

        reference_count = sum(1 for ref in references if ref in getting_started_content.lower())
        assert reference_count >= 2, "Getting started should reference other documentation"

        print("✅ Documentation structure is logical and interconnected")

    def test_code_examples_are_complete(self):
        """Test that code examples in documentation are complete and runnable."""
        getting_started_path = Path("external-docs/getting-started.md")
        content = getting_started_path.read_text()

        # Extract Python code blocks
        import re

        code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

        assert len(code_blocks) >= 3, "Should have multiple Python code examples"

        # Check that examples include essential imports
        has_saplings_import = any(
            "from saplings import" in block or "import saplings" in block for block in code_blocks
        )
        assert has_saplings_import, "Should show how to import Saplings"

        # Check for async examples
        has_async_example = any("async def" in block and "await" in block for block in code_blocks)
        assert has_async_example, "Should include async/await examples"

        # Check for AgentConfig usage
        has_agent_config = any("AgentConfig" in block for block in code_blocks)
        assert has_agent_config, "Should show AgentConfig usage"

        print("✅ Code examples are complete and demonstrate key concepts")

    def test_documentation_covers_user_journey(self):
        """Test that documentation covers the complete user journey."""
        # Define the user journey stages
        journey_stages = {
            "installation.md": ["install", "setup", "dependencies"],
            "getting-started.md": ["first agent", "basic usage", "quick start"],
            "quick-start-tutorial.md": ["tutorial", "step by step", "hands-on"],
            "troubleshooting.md": ["problems", "errors", "solutions"],
        }

        for doc_file, expected_content in journey_stages.items():
            doc_path = Path(f"external-docs/{doc_file}")
            assert doc_path.exists(), f"{doc_file} should exist"

            content = doc_path.read_text().lower()

            for expected in expected_content:
                assert expected in content, f"{doc_file} should cover '{expected}'"

        print("✅ Documentation covers complete user journey")

    def test_documentation_is_beginner_friendly(self):
        """Test that documentation is accessible to beginners."""
        getting_started_path = Path("external-docs/getting-started.md")
        content = getting_started_path.read_text()

        # Check for beginner-friendly elements
        beginner_elements = [
            "What is",  # Explanations
            "Quick Start",  # Fast path to success
            "5 minutes",  # Time expectations
            "Step 1",  # Clear steps
            "Prerequisites",  # Requirements
            "Next Steps",  # Progression
            "Common",  # Common patterns/issues
            "Example",  # Examples
        ]

        found_elements = sum(
            1 for element in beginner_elements if element.lower() in content.lower()
        )

        assert (
            found_elements >= 6
        ), f"Should have beginner-friendly elements, found {found_elements}"

        # Check for encouraging language
        encouraging_phrases = ["Congratulations", "You've created", "Ready to", "🎉", "✅"]

        found_encouragement = sum(1 for phrase in encouraging_phrases if phrase in content)

        assert found_encouragement >= 2, "Should include encouraging language"

        print("✅ Documentation is beginner-friendly")

    def test_documentation_summary(self):
        """Test that we can summarize the documentation completeness."""
        print("\n=== Documentation Summary ===")

        docs_created = []
        external_docs_path = Path("external-docs")

        if external_docs_path.exists():
            for doc_file in external_docs_path.glob("*.md"):
                docs_created.append(doc_file.name)
                print(f"  ✅ {doc_file.name}")

        expected_docs = [
            "getting-started.md",
            "quick-start-tutorial.md",
            "troubleshooting.md",
            "installation.md",
        ]

        missing_docs = [doc for doc in expected_docs if doc not in docs_created]

        print(f"\nDocumentation files: {len(docs_created)}/{len(expected_docs)}")

        if missing_docs:
            print(f"Missing: {missing_docs}")
        else:
            print("✅ All essential documentation exists")

        # Check total content
        total_lines = 0
        for doc in docs_created:
            doc_path = external_docs_path / doc
            lines = len(doc_path.read_text().splitlines())
            total_lines += lines
            print(f"  - {doc}: {lines} lines")

        print(f"\nTotal documentation: {total_lines} lines")

        assert len(docs_created) >= 4, f"Expected at least 4 docs, got {len(docs_created)}"
        assert total_lines >= 800, f"Expected substantial documentation, got {total_lines} lines"


if __name__ == "__main__":
    # Run the tests when script is executed directly
    pytest.main([__file__, "-v"])
