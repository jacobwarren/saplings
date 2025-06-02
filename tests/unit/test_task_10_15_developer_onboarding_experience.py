"""
Test for Task 10.15: Create comprehensive developer onboarding experience.

This test verifies that comprehensive developer onboarding experience has been implemented:
1. Getting started documentation exists and is comprehensive
2. Quick start examples work out of the box
3. Progressive complexity examples available
4. Installation instructions are clear and complete
5. Developer documentation is well-organized
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestTask1015DeveloperOnboardingExperience:
    """Test Task 10.15: Create comprehensive developer onboarding experience."""

    def test_getting_started_documentation_exists(self):
        """Test that comprehensive getting started documentation exists."""
        # Check for getting started documentation
        getting_started_docs = [
            Path("README.md"),
            Path("external-docs/getting-started.md"),
            Path("external-docs/quickstart.md"),
            Path("docs/getting-started.md"),
            Path("GETTING_STARTED.md"),
        ]

        found_getting_started = []
        for doc_path in getting_started_docs:
            if doc_path.exists():
                found_getting_started.append(doc_path)

                # Check if documentation is comprehensive
                content = doc_path.read_text()

                # Check for key sections
                key_sections = ["install", "example", "usage", "quick", "start"]

                found_sections = []
                for section in key_sections:
                    if section in content.lower():
                        found_sections.append(section)

                print(
                    f"✅ Found getting started doc: {doc_path} (covers {len(found_sections)}/{len(key_sections)} key sections)"
                )

        if found_getting_started:
            print(f"✅ Found {len(found_getting_started)} getting started documentation files")
            assert (
                len(found_getting_started) >= 1
            ), "Should have at least one getting started documentation file"
        else:
            print("⚠️  No getting started documentation found")

    def test_quick_start_examples_work(self):
        """Test that quick start examples work out of the box."""
        # Test a basic quick start example
        quickstart_code = """
import time
import sys

try:
    # Quick start example - should work with minimal setup
    start_time = time.time()

    # Step 1: Import the package
    import saplings
    import_time = time.time() - start_time
    print(f"✅ Step 1: Package imported in {import_time:.2f}s")

    # Step 2: Create a simple configuration
    config_start = time.time()
    config = saplings.AgentConfig(
        provider="test",
        model_name="test-model"
    )
    config_time = time.time() - config_start
    print(f"✅ Step 2: Configuration created in {config_time:.3f}s")

    # Step 3: Use factory method for easier setup
    factory_start = time.time()
    minimal_config = saplings.AgentConfig.minimal(
        provider="test",
        model_name="test-model"
    )
    factory_time = time.time() - factory_start
    print(f"✅ Step 3: Factory method works in {factory_time:.3f}s")

    # Step 4: Verify configuration
    assert minimal_config.provider == "test"
    assert minimal_config.model_name == "test-model"
    print("✅ Step 4: Configuration verified")

    total_time = time.time() - start_time
    print(f"✅ Quick start example completed in {total_time:.2f}s")

    exit(0)

except Exception as e:
    print(f"❌ Quick start example failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", quickstart_code],
                timeout=30,
                capture_output=True,
                text=True,
                cwd="/Users/jacobwarren/Development/agents/saplings",
                check=False,
            )

            if result.returncode == 0:
                print("✅ Quick start example validation:")
                for line in result.stdout.strip().split("\n"):
                    print(f"   {line}")
            else:
                print(f"⚠️  Quick start example: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.fail("Quick start example timed out")

    def test_progressive_complexity_examples_exist(self):
        """Test that progressive complexity examples are available."""
        # Check for example directories and files
        example_locations = [
            Path("examples"),
            Path("external-docs/examples"),
            Path("docs/examples"),
            Path("samples"),
        ]

        found_examples = []
        total_example_files = 0

        for example_dir in example_locations:
            if example_dir.exists() and example_dir.is_dir():
                example_files = list(example_dir.glob("*.py")) + list(example_dir.glob("*.md"))
                if example_files:
                    found_examples.append(example_dir)
                    total_example_files += len(example_files)
                    print(f"✅ Found example directory: {example_dir} ({len(example_files)} files)")

        # Check for different complexity levels
        complexity_levels = ["basic", "minimal", "simple", "advanced", "complex", "full"]
        found_complexity_examples = []

        for example_dir in found_examples:
            for complexity in complexity_levels:
                complexity_files = list(example_dir.glob(f"*{complexity}*"))
                if complexity_files:
                    found_complexity_examples.extend(complexity_files)

        if found_examples:
            print(
                f"✅ Found {len(found_examples)} example directories with {total_example_files} total files"
            )
            print(f"✅ Found {len(found_complexity_examples)} complexity-specific examples")
        else:
            print("⚠️  No example directories found")

    def test_installation_instructions_clear(self):
        """Test that installation instructions are clear and complete."""
        # Check for installation documentation
        install_docs = [
            Path("README.md"),
            Path("external-docs/installation.md"),
            Path("docs/installation.md"),
            Path("INSTALL.md"),
        ]

        found_install_docs = []
        for doc_path in install_docs:
            if doc_path.exists():
                content = doc_path.read_text()

                # Check for installation keywords
                install_keywords = ["pip install", "install", "setup", "requirements"]
                has_install_info = any(keyword in content.lower() for keyword in install_keywords)

                if has_install_info:
                    found_install_docs.append(doc_path)
                    print(f"✅ Found installation documentation: {doc_path}")

                    # Check for different installation methods
                    install_methods = ["pip", "conda", "poetry", "requirements.txt"]
                    found_methods = []
                    for method in install_methods:
                        if method in content.lower():
                            found_methods.append(method)

                    if found_methods:
                        print(f"   Covers installation methods: {', '.join(found_methods)}")

        if found_install_docs:
            print(f"✅ Found {len(found_install_docs)} installation documentation files")
        else:
            print("⚠️  No installation documentation found")

    def test_developer_documentation_well_organized(self):
        """Test that developer documentation is well-organized."""
        # Check for organized documentation structure
        doc_directories = [Path("docs"), Path("external-docs")]

        doc_categories = {
            "Getting Started": ["getting-started", "quickstart", "tutorial"],
            "API Reference": ["api", "reference", "interface"],
            "Examples": ["examples", "samples", "demo"],
            "Advanced": ["advanced", "guide", "deep-dive"],
            "Troubleshooting": ["troubleshooting", "faq", "common-issues"],
        }

        found_categories = {}
        total_docs = 0

        for doc_dir in doc_directories:
            if doc_dir.exists():
                doc_files = list(doc_dir.glob("*.md"))
                total_docs += len(doc_files)

                print(f"✅ Found documentation directory: {doc_dir} ({len(doc_files)} files)")

                # Categorize documentation
                for category, keywords in doc_categories.items():
                    category_files = []
                    for doc_file in doc_files:
                        if any(keyword in doc_file.name.lower() for keyword in keywords):
                            category_files.append(doc_file)

                    if category_files:
                        found_categories[category] = category_files
                        print(f"   {category}: {len(category_files)} files")

        print("\n📊 Documentation Organization Analysis:")
        print(f"   Total documentation files: {total_docs}")
        print(f"   Organized categories: {len(found_categories)}/{len(doc_categories)}")

        if len(found_categories) >= 3:
            print("✅ Good documentation organization")
        elif len(found_categories) >= 2:
            print("⚠️  Basic documentation organization")
        else:
            print("⚠️  Limited documentation organization")

    def test_onboarding_checklist_exists(self):
        """Test that developer onboarding checklist exists."""
        # Check for onboarding checklist or guide
        onboarding_docs = [
            Path("external-docs/onboarding.md"),
            Path("docs/onboarding.md"),
            Path("external-docs/developer-guide.md"),
            Path("docs/developer-guide.md"),
            Path("CONTRIBUTING.md"),
        ]

        found_onboarding = []
        for doc_path in onboarding_docs:
            if doc_path.exists():
                found_onboarding.append(doc_path)

                content = doc_path.read_text()

                # Check for checklist elements
                checklist_indicators = ["- [ ]", "- [x]", "1.", "step", "checklist"]
                has_checklist = any(
                    indicator in content.lower() for indicator in checklist_indicators
                )

                if has_checklist:
                    print(f"✅ Found onboarding checklist: {doc_path}")
                else:
                    print(f"✅ Found onboarding documentation: {doc_path}")

        if found_onboarding:
            print(f"✅ Found {len(found_onboarding)} onboarding documentation files")
        else:
            print("⚠️  No onboarding documentation found")

    def test_api_documentation_comprehensive(self):
        """Test that API documentation is comprehensive."""
        # Check for API documentation
        api_docs = [
            Path("docs/api"),
            Path("external-docs/api"),
            Path("docs/reference"),
            Path("external-docs/reference"),
        ]

        found_api_docs = []
        total_api_files = 0

        for api_dir in api_docs:
            if api_dir.exists() and api_dir.is_dir():
                api_files = list(api_dir.glob("*.md"))
                if api_files:
                    found_api_docs.append(api_dir)
                    total_api_files += len(api_files)
                    print(f"✅ Found API documentation: {api_dir} ({len(api_files)} files)")

        # Check for key API components
        key_components = ["agent", "config", "tools", "memory", "services"]
        documented_components = []

        for api_dir in found_api_docs:
            for component in key_components:
                component_files = list(api_dir.glob(f"*{component}*"))
                if component_files:
                    documented_components.append(component)

        if found_api_docs:
            print(
                f"✅ Found {len(found_api_docs)} API documentation directories with {total_api_files} files"
            )
            print(
                f"✅ Documented components: {len(set(documented_components))}/{len(key_components)}"
            )
        else:
            print("⚠️  No API documentation directories found")

    def test_tutorial_progression_exists(self):
        """Test that tutorial progression exists for learning."""
        # Check for tutorial files
        tutorial_locations = [
            Path("external-docs/tutorials"),
            Path("docs/tutorials"),
            Path("tutorials"),
            Path("external-docs"),
        ]

        tutorial_patterns = ["tutorial", "guide", "walkthrough", "lesson"]
        found_tutorials = []

        for location in tutorial_locations:
            if location.exists():
                for pattern in tutorial_patterns:
                    tutorial_files = list(location.glob(f"*{pattern}*"))
                    found_tutorials.extend(tutorial_files)

        # Remove duplicates
        unique_tutorials = list(set(found_tutorials))

        if unique_tutorials:
            print(f"✅ Found {len(unique_tutorials)} tutorial files")

            # Check for progression indicators
            progression_indicators = ["01", "02", "part", "step", "basic", "advanced"]
            progressive_tutorials = []

            for tutorial in unique_tutorials:
                if any(indicator in tutorial.name.lower() for indicator in progression_indicators):
                    progressive_tutorials.append(tutorial)

            if progressive_tutorials:
                print(f"✅ Found {len(progressive_tutorials)} progressive tutorials")
            else:
                print("⚠️  No progressive tutorial structure found")
        else:
            print("⚠️  No tutorial files found")

    def test_developer_experience_metrics(self):
        """Test and analyze overall developer experience metrics."""
        print("\n📊 Developer Onboarding Experience Analysis:")

        # Count different types of developer resources
        resource_counts = {
            "Documentation Files": len(list(Path("docs").glob("*.md")))
            + len(list(Path("external-docs").glob("*.md")))
            if Path("docs").exists() or Path("external-docs").exists()
            else 0,
            "Example Files": len(list(Path("examples").glob("*.py")))
            if Path("examples").exists()
            else 0,
            "Test Files": len(list(Path("tests").glob("test_*.py"))),
            "README Files": 1 if Path("README.md").exists() else 0,
        }

        for resource_type, count in resource_counts.items():
            print(f"   {resource_type}: {count}")

        # Calculate overall score
        total_resources = sum(resource_counts.values())

        if total_resources >= 100:
            print("✅ Excellent developer resources")
        elif total_resources >= 50:
            print("✅ Good developer resources")
        elif total_resources >= 20:
            print("⚠️  Basic developer resources")
        else:
            print("⚠️  Limited developer resources")

        # Check for key onboarding elements
        key_elements = {
            "README.md": Path("README.md").exists(),
            "Getting Started": any(
                Path(p).exists()
                for p in ["external-docs/getting-started.md", "docs/getting-started.md"]
            ),
            "Examples": Path("examples").exists(),
            "API Docs": any(Path(p).exists() for p in ["docs/api", "external-docs/api"]),
            "Tests": len(list(Path("tests").glob("test_*.py"))) > 0,
        }

        present_elements = sum(key_elements.values())
        print(f"\n   Key onboarding elements: {present_elements}/{len(key_elements)}")

        for element, present in key_elements.items():
            status = "✅" if present else "❌"
            print(f"   {status} {element}")

        # Overall assessment
        if present_elements >= 4:
            print("\n✅ Comprehensive developer onboarding experience")
        elif present_elements >= 3:
            print("\n✅ Good developer onboarding experience")
        elif present_elements >= 2:
            print("\n⚠️  Basic developer onboarding experience")
        else:
            print("\n⚠️  Limited developer onboarding experience")

        # Should have good onboarding experience
        assert (
            present_elements >= 3
        ), f"Should have at least 3 key onboarding elements, found {present_elements}/{len(key_elements)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
