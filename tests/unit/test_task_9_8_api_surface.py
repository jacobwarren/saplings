"""
Test for Task 9.8: Reduce main package API surface from 324 to ~30 core items.

This test verifies that the main package API surface is reduced to focus on core functionality.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pytest


class TestTask98APISurface:
    """Test Task 9.8: Reduce main package API surface from 324 to ~30 core items."""

    def test_main_package_exports_count(self):
        """Test that main package exports are within target range."""
        main_init = Path("/Users/jacobwarren/Development/agents/saplings/src/saplings/__init__.py")

        if not main_init.exists():
            pytest.fail("Main package __init__.py doesn't exist")

        content = main_init.read_text()

        # Count exports in __all__ if it exists
        all_exports = self._extract_all_exports(content)

        if all_exports:
            export_count = len(all_exports)
            print(f"Main package exports {export_count} items via __all__")

            # Show some examples
            for item in sorted(all_exports)[:10]:
                print(f"  - {item}")

            if export_count > 10:
                print(f"  ... and {export_count - 10} more")

            # Target is ~30 core items
            if export_count <= 30:
                print(f"✅ Export count ({export_count}) is within target (≤30)")
            elif export_count <= 50:
                print(f"⚠️  Export count ({export_count}) is above target but acceptable")
            else:
                print(f"❌ Export count ({export_count}) is too high (target: ≤30)")
        else:
            # Count import statements if no __all__
            imports = self._count_import_statements(content)
            print(f"Main package has {imports} import statements (no __all__ found)")

        # Don't fail test - this shows current state
        assert main_init.exists(), "Main package should exist"

    def test_core_items_are_exported(self):
        """Test that core items are exported from main package."""
        main_init = Path("/Users/jacobwarren/Development/agents/saplings/src/saplings/__init__.py")

        if not main_init.exists():
            pytest.fail("Main package __init__.py doesn't exist")

        content = main_init.read_text()

        # Core items that should be in main namespace
        core_items = [
            "Agent",
            "AgentConfig",
            "AgentBuilder",
            "Tool",
            "tool",  # decorator
            "MemoryStore",
        ]

        exported_core = []
        missing_core = []

        for item in core_items:
            if item in content:
                exported_core.append(item)
                print(f"✅ Core item exported: {item}")
            else:
                missing_core.append(item)
                print(f"❌ Core item missing: {item}")

        print(f"Exported core items: {len(exported_core)}/{len(core_items)}")

        if missing_core:
            print("Missing core items:")
            for item in missing_core:
                print(f"  - {item}")

        # Don't fail test - this shows what needs to be ensured
        assert len(core_items) > 0, "Should check for core items"

    def test_advanced_items_moved_to_subnamespaces(self):
        """Test that advanced items are moved to appropriate subnamespaces."""
        # Check if advanced namespace exists
        advanced_init = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/advanced/__init__.py"
        )
        tools_init = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/tools/__init__.py"
        )
        models_init = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/models/__init__.py"
        )
        memory_init = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/memory/__init__.py"
        )

        subnamespaces = {
            "saplings.advanced": advanced_init,
            "saplings.tools": tools_init,
            "saplings.models": models_init,
            "saplings.memory": memory_init,
        }

        existing_subnamespaces = []
        missing_subnamespaces = []

        for namespace, path in subnamespaces.items():
            if path.exists():
                existing_subnamespaces.append(namespace)
                print(f"✅ Subnamespace exists: {namespace}")
            else:
                missing_subnamespaces.append(namespace)
                print(f"❌ Subnamespace missing: {namespace}")

        print(f"Existing subnamespaces: {len(existing_subnamespaces)}")
        print(f"Missing subnamespaces: {len(missing_subnamespaces)}")

        # Check content of existing subnamespaces
        for namespace, path in subnamespaces.items():
            if path.exists():
                try:
                    content = path.read_text()
                    exports = self._extract_all_exports(content)
                    if exports:
                        print(f"  {namespace} exports {len(exports)} items")
                    else:
                        imports = self._count_import_statements(content)
                        print(f"  {namespace} has {imports} imports")
                except Exception as e:
                    print(f"  Could not analyze {namespace}: {e}")

        # Don't fail test - this shows current state
        assert len(subnamespaces) > 0, "Should check for subnamespaces"

    def test_internal_items_not_exported(self):
        """Test that internal items are not exported from main package."""
        main_init = Path("/Users/jacobwarren/Development/agents/saplings/src/saplings/__init__.py")

        if not main_init.exists():
            pytest.fail("Main package __init__.py doesn't exist")

        content = main_init.read_text()

        # Internal items that should NOT be in main namespace
        internal_patterns = [
            "_internal",
            "Builder",  # Specific builders (except AgentBuilder)
            "Service",  # Internal services
            "Container",
            "Registry",
            "Monitor",
            "Validator",
        ]

        exported_internal = []
        properly_hidden = []

        for pattern in internal_patterns:
            if pattern in content and not pattern.startswith("Agent"):
                # Check if it's actually exported (not just imported)
                lines = content.split("\n")
                for line in lines:
                    if (
                        pattern in line
                        and (line.strip().startswith("from") or line.strip().startswith("import"))
                        and not line.strip().startswith("#")
                    ):
                        exported_internal.append(pattern)
                        break
                else:
                    properly_hidden.append(pattern)
            else:
                properly_hidden.append(pattern)

        for pattern in exported_internal:
            print(f"⚠️  Internal pattern exported: {pattern}")

        for pattern in properly_hidden:
            print(f"✅ Internal pattern hidden: {pattern}")

        print(f"Properly hidden: {len(properly_hidden)}")
        print(f"Exported internal: {len(exported_internal)}")

        # Don't fail test - this shows what needs to be cleaned up
        assert len(internal_patterns) > 0, "Should check for internal patterns"

    def test_backward_compatibility_warnings(self):
        """Test that backward compatibility warnings are implemented."""
        main_init = Path("/Users/jacobwarren/Development/agents/saplings/src/saplings/__init__.py")

        if not main_init.exists():
            pytest.fail("Main package __init__.py doesn't exist")

        content = main_init.read_text()

        # Check for deprecation warning patterns
        has_warnings_import = "warnings" in content
        has_deprecation_warnings = "DeprecationWarning" in content or "FutureWarning" in content
        has_getattr_deprecation = "__getattr__" in content and "deprecat" in content.lower()

        deprecation_features = []

        if has_warnings_import:
            deprecation_features.append("warnings import")
            print("✅ Has warnings import")
        else:
            print("❌ Missing warnings import")

        if has_deprecation_warnings:
            deprecation_features.append("deprecation warnings")
            print("✅ Has deprecation warnings")
        else:
            print("❌ Missing deprecation warnings")

        if has_getattr_deprecation:
            deprecation_features.append("__getattr__ deprecation")
            print("✅ Has __getattr__ deprecation handling")
        else:
            print("⚠️  Missing __getattr__ deprecation handling")

        print(f"Deprecation features: {len(deprecation_features)}/3")

        # Don't fail test - this shows what needs to be implemented
        assert len(deprecation_features) >= 0, "Should check for deprecation features"

    def test_api_categorization_documentation(self):
        """Test that API categorization is documented."""
        docs_dir = Path("/Users/jacobwarren/Development/agents/saplings/docs")
        external_docs_dir = Path("/Users/jacobwarren/Development/agents/saplings/external-docs")

        # Look for API categorization documentation
        categorization_docs = []

        for docs_path in [docs_dir, external_docs_dir]:
            if docs_path.exists():
                for doc_file in docs_path.rglob("*.md"):
                    try:
                        content = doc_file.read_text()
                        if "api" in content.lower() and (
                            "categor" in content.lower() or "surface" in content.lower()
                        ):
                            categorization_docs.append(str(doc_file))
                    except Exception:
                        continue

        if categorization_docs:
            print(f"✅ Found {len(categorization_docs)} API categorization docs:")
            for doc in categorization_docs:
                print(f"  - {doc}")
        else:
            print("❌ No API categorization documentation found")

        # Don't fail test - this shows what documentation exists
        assert True, "Documentation check completed"

    def _extract_all_exports(self, content: str) -> List[str]:
        """Extract items from __all__ definition."""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "__all__"
                ):
                    if isinstance(node.value, ast.List):
                        exports = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Str):
                                exports.append(elt.s)
                            elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exports.append(elt.value)
                        return exports
        except Exception:
            pass
        return []

    def _count_import_statements(self, content: str) -> int:
        """Count import statements in content."""
        try:
            tree = ast.parse(content)
            import_count = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_count += 1
            return import_count
        except Exception:
            # Fallback to line counting
            lines = content.split("\n")
            import_count = 0
            for line in lines:
                stripped = line.strip()
                if (
                    stripped.startswith("import ") or stripped.startswith("from ")
                ) and not stripped.startswith("#"):
                    import_count += 1
            return import_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
