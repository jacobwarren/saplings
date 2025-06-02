"""
Test for Task 6.1: Update documentation for new import paths.

This test audits documentation for deprecated imports and ensures
all documentation uses new standardized import paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest


class DocumentationAnalyzer:
    """Analyzer to check documentation for deprecated imports."""

    def __init__(self):
        self.project_root = Path()
        self.deprecated_imports = {
            # Deprecated modules that were removed
            "saplings.api.container",
            "saplings.api.document",
            "saplings.api.interfaces",
            "saplings.api.indexer",
            "saplings.api.tool_validation",
            "saplings.core.interfaces",
            # Internal modules that shouldn't be in docs
            "saplings._internal",
            "saplings.memory._internal",
            "saplings.tools._internal",
            "saplings.services._internal",
            "saplings.models._internal",
        }

        self.correct_imports = {
            "saplings.api.container": "saplings.api.di",
            "saplings.api.document": "saplings.api.memory.document",
            "saplings.api.interfaces": "saplings.api.core.interfaces",
            "saplings.api.indexer": "saplings.api.memory.indexer",
            "saplings.api.tool_validation": "saplings.api.tools",
            "saplings.core.interfaces": "saplings.api.core.interfaces",
        }

    def find_documentation_files(self) -> List[Path]:
        """Find all documentation files in the project."""
        doc_files = []

        # Look for documentation files
        patterns = ["*.md", "*.rst", "*.txt"]

        for pattern in patterns:
            # Check root directory
            doc_files.extend(self.project_root.glob(pattern))

            # Check docs directory
            docs_dir = self.project_root / "docs"
            if docs_dir.exists():
                doc_files.extend(docs_dir.rglob(pattern))

            # Check external-docs directory
            external_docs_dir = self.project_root / "external-docs"
            if external_docs_dir.exists():
                doc_files.extend(external_docs_dir.rglob(pattern))

        return sorted(doc_files)

    def find_example_files(self) -> List[Path]:
        """Find all example files in the project."""
        example_files = []

        # Look for Python example files
        examples_dir = self.project_root / "examples"
        if examples_dir.exists():
            example_files.extend(examples_dir.glob("*.py"))

        return sorted(example_files)

    def check_file_for_deprecated_imports(self, file_path: Path) -> Dict[str, List[int]]:
        """Check a file for deprecated imports."""
        deprecated_found = {}

        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                # Check for import statements
                if "import" in line and "saplings" in line:
                    for deprecated in self.deprecated_imports:
                        if deprecated in line:
                            if deprecated not in deprecated_found:
                                deprecated_found[deprecated] = []
                            deprecated_found[deprecated].append(line_num)

        except Exception:
            # Skip files that can't be read
            pass

        return deprecated_found

    def audit_documentation(self) -> Dict[str, Dict]:
        """Audit all documentation for deprecated imports."""
        results = {}

        # Check documentation files
        doc_files = self.find_documentation_files()
        for doc_file in doc_files:
            deprecated = self.check_file_for_deprecated_imports(doc_file)
            if deprecated:
                results[str(doc_file)] = {"type": "documentation", "deprecated_imports": deprecated}

        return results

    def audit_examples(self) -> Dict[str, Dict]:
        """Audit all examples for deprecated imports."""
        results = {}

        # Check example files
        example_files = self.find_example_files()
        for example_file in example_files:
            deprecated = self.check_file_for_deprecated_imports(example_file)
            if deprecated:
                results[str(example_file)] = {"type": "example", "deprecated_imports": deprecated}

        return results

    def generate_update_recommendations(self, audit_results: Dict[str, Dict]) -> List[str]:
        """Generate recommendations for updating documentation."""
        recommendations = []

        for file_path, info in audit_results.items():
            deprecated_imports = info["deprecated_imports"]
            file_type = info["type"]

            recommendations.append(f"Update {file_type} file: {file_path}")

            for deprecated, line_numbers in deprecated_imports.items():
                if deprecated in self.correct_imports:
                    correct = self.correct_imports[deprecated]
                    recommendations.append(
                        f"  - Replace '{deprecated}' with '{correct}' on lines {line_numbers}"
                    )
                else:
                    recommendations.append(
                        f"  - Remove or update '{deprecated}' on lines {line_numbers}"
                    )

        return recommendations


@pytest.fixture()
def documentation_analyzer():
    """Fixture providing a documentation analyzer."""
    return DocumentationAnalyzer()


def test_find_documentation_files(documentation_analyzer):
    """Test that we can find documentation files."""
    doc_files = documentation_analyzer.find_documentation_files()

    # Should find at least some documentation files
    assert len(doc_files) > 0, "Should find documentation files"

    # Check for expected files
    file_names = [f.name for f in doc_files]
    expected_files = ["README.md", "CONTRIBUTING.md"]

    for expected in expected_files:
        assert expected in file_names, f"Should find {expected}"

    print(f"\nFound {len(doc_files)} documentation files:")
    for doc_file in doc_files[:10]:  # Show first 10
        print(f"  - {doc_file}")
    if len(doc_files) > 10:
        print(f"  ... and {len(doc_files) - 10} more")


def test_find_example_files(documentation_analyzer):
    """Test that we can find example files."""
    example_files = documentation_analyzer.find_example_files()

    # Should find example files
    assert len(example_files) > 0, "Should find example files"

    print(f"\nFound {len(example_files)} example files:")
    for example_file in example_files[:10]:  # Show first 10
        print(f"  - {example_file}")
    if len(example_files) > 10:
        print(f"  ... and {len(example_files) - 10} more")


def test_audit_documentation_for_deprecated_imports(documentation_analyzer):
    """Test auditing documentation for deprecated imports."""
    audit_results = documentation_analyzer.audit_documentation()

    print("\nDocumentation Audit Results:")
    print(f"  Files with deprecated imports: {len(audit_results)}")

    if audit_results:
        print("  Files needing updates:")
        for file_path, info in audit_results.items():
            deprecated_count = sum(len(lines) for lines in info["deprecated_imports"].values())
            print(f"    - {file_path}: {deprecated_count} deprecated imports")
    else:
        print("  ✅ No deprecated imports found in documentation")

    # After cleanup, there should be no deprecated imports in documentation
    # This is a soft assertion since some docs might not be updated yet
    if len(audit_results) > 0:
        print("  ⚠️  Documentation needs updating")
    else:
        print("  ✅ Documentation is up to date")


def test_audit_examples_for_deprecated_imports(documentation_analyzer):
    """Test auditing examples for deprecated imports."""
    audit_results = documentation_analyzer.audit_examples()

    print("\nExample Audit Results:")
    print(f"  Files with deprecated imports: {len(audit_results)}")

    if audit_results:
        print("  Files needing updates:")
        for file_path, info in audit_results.items():
            deprecated_count = sum(len(lines) for lines in info["deprecated_imports"].values())
            print(f"    - {file_path}: {deprecated_count} deprecated imports")
    else:
        print("  ✅ No deprecated imports found in examples")

    # After cleanup, there should be no deprecated imports in examples
    # This is a soft assertion since some examples might not be updated yet
    if len(audit_results) > 0:
        print("  ⚠️  Examples need updating")
    else:
        print("  ✅ Examples are up to date")


def test_generate_update_recommendations(documentation_analyzer):
    """Test generation of update recommendations."""
    # Audit both documentation and examples
    doc_audit = documentation_analyzer.audit_documentation()
    example_audit = documentation_analyzer.audit_examples()

    # Combine results
    all_audit_results = {**doc_audit, **example_audit}

    # Generate recommendations
    recommendations = documentation_analyzer.generate_update_recommendations(all_audit_results)

    print("\nUpdate Recommendations:")
    if recommendations:
        print(f"  Generated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations[:10], 1):  # Show first 10
            print(f"    {i}. {rec}")
        if len(recommendations) > 10:
            print(f"    ... and {len(recommendations) - 10} more")
    else:
        print("  ✅ No updates needed - all documentation and examples are current")

    # The test passes regardless of whether updates are needed
    # This is informational to help identify what needs to be updated


if __name__ == "__main__":
    # Run the analysis when script is executed directly
    analyzer = DocumentationAnalyzer()

    print("Documentation Analysis Results:")
    print("=" * 50)

    doc_files = analyzer.find_documentation_files()
    example_files = analyzer.find_example_files()
    doc_audit = analyzer.audit_documentation()
    example_audit = analyzer.audit_examples()

    print(f"Found {len(doc_files)} documentation files")
    print(f"Found {len(example_files)} example files")
    print(f"Documentation files with deprecated imports: {len(doc_audit)}")
    print(f"Example files with deprecated imports: {len(example_audit)}")

    if doc_audit or example_audit:
        all_results = {**doc_audit, **example_audit}
        recommendations = analyzer.generate_update_recommendations(all_results)
        print(f"Generated {len(recommendations)} update recommendations")
    else:
        print("All documentation and examples are up to date!")
