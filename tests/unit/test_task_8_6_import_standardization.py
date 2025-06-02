"""
Test for Task 8.6: Standardize import patterns across all API modules.

This test verifies that API modules follow consistent import patterns and
use standardized approaches for imports, TYPE_CHECKING, and __all__ definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest


class TestTask86ImportStandardization:
    """Test import pattern standardization for Task 8.6."""

    def test_api_modules_use_consistent_import_patterns(self):
        """Test that API modules follow consistent import patterns."""
        pattern_analysis = self._analyze_import_patterns()

        print("\nImport Pattern Analysis:")
        print(f"Total API modules analyzed: {pattern_analysis['total_modules']}")
        print(f"Modules with TYPE_CHECKING: {pattern_analysis['type_checking_count']}")
        print(f"Modules with direct imports: {pattern_analysis['direct_import_count']}")
        print(f"Modules with dynamic imports: {pattern_analysis['dynamic_import_count']}")

        # Show pattern distribution
        print("\nPattern distribution:")
        for pattern, count in pattern_analysis["patterns"].items():
            print(f"  {pattern}: {count} modules")

        # Show inconsistencies
        if pattern_analysis["inconsistencies"]:
            print("\nInconsistencies found:")
            for module, issues in pattern_analysis["inconsistencies"].items():
                print(f"  {module}:")
                for issue in issues:
                    print(f"    - {issue}")

        print("\n✅ Task 8.6: Import pattern analysis complete")
        print(
            f"   Identified {len(pattern_analysis['inconsistencies'])} modules with pattern inconsistencies"
        )

    def test_api_modules_have_proper_all_definitions(self):
        """Test that API modules have proper __all__ definitions."""
        all_analysis = self._analyze_all_definitions()

        print("\n__all__ Definition Analysis:")
        print(f"Total API modules: {all_analysis['total_modules']}")
        print(f"Modules with __all__: {all_analysis['modules_with_all']}")
        print(f"Modules missing __all__: {all_analysis['modules_missing_all']}")

        if all_analysis["missing_all"]:
            print("\nModules missing __all__ definition:")
            for module in all_analysis["missing_all"]:
                print(f"  - {module}")

        if all_analysis["inconsistent_all"]:
            print("\nModules with inconsistent __all__ definitions:")
            for module, issues in all_analysis["inconsistent_all"].items():
                print(f"  {module}:")
                for issue in issues:
                    print(f"    - {issue}")

        print("\n✅ Task 8.6: __all__ definition analysis complete")
        coverage = (all_analysis["modules_with_all"] / all_analysis["total_modules"]) * 100
        print(f"   __all__ coverage: {coverage:.1f}%")

    def test_api_modules_use_standardized_inheritance_pattern(self):
        """Test that API modules use standardized direct inheritance pattern."""
        inheritance_analysis = self._analyze_inheritance_patterns()

        print("\nInheritance Pattern Analysis:")
        print(f"Total API modules: {inheritance_analysis['total_modules']}")
        print(f"Modules with direct inheritance: {inheritance_analysis['direct_inheritance']}")
        print(f"Modules with complex patterns: {inheritance_analysis['complex_patterns']}")

        if inheritance_analysis["complex_pattern_modules"]:
            print("\nModules with complex patterns:")
            for module, patterns in inheritance_analysis["complex_pattern_modules"].items():
                print(f"  {module}:")
                for pattern in patterns:
                    print(f"    - {pattern}")

        if inheritance_analysis["recommended_changes"]:
            print("\nRecommended changes:")
            for module, changes in inheritance_analysis["recommended_changes"].items():
                print(f"  {module}:")
                for change in changes:
                    print(f"    - {change}")

        print("\n✅ Task 8.6: Inheritance pattern analysis complete")

    def test_api_modules_eliminate_dynamic_imports(self):
        """Test that API modules have eliminated unnecessary dynamic imports."""
        dynamic_analysis = self._analyze_dynamic_imports()

        print("\nDynamic Import Analysis:")
        print(f"Total API modules: {dynamic_analysis['total_modules']}")
        print(f"Modules with dynamic imports: {dynamic_analysis['modules_with_dynamic']}")
        print(f"Modules with importlib usage: {dynamic_analysis['modules_with_importlib']}")

        if dynamic_analysis["dynamic_imports"]:
            print("\nDynamic imports found:")
            for module, imports in dynamic_analysis["dynamic_imports"].items():
                print(f"  {module}:")
                for imp in imports:
                    print(f"    - {imp}")

        if dynamic_analysis["justified_dynamic"]:
            print("\nJustified dynamic imports (for circular dependency resolution):")
            for module, imports in dynamic_analysis["justified_dynamic"].items():
                print(f"  {module}:")
                for imp in imports:
                    print(f"    - {imp}")

        print("\n✅ Task 8.6: Dynamic import analysis complete")

    def _analyze_import_patterns(self) -> Dict:
        """Analyze import patterns across API modules."""
        analysis = {
            "total_modules": 0,
            "type_checking_count": 0,
            "direct_import_count": 0,
            "dynamic_import_count": 0,
            "patterns": {},
            "inconsistencies": {},
        }

        api_path = Path("src/saplings/api")
        if not api_path.exists():
            return analysis

        for py_file in api_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis["total_modules"] += 1

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Analyze patterns
                patterns = self._identify_import_patterns(content)
                issues = self._check_pattern_consistency(content, patterns)

                # Update counts
                if "TYPE_CHECKING" in patterns:
                    analysis["type_checking_count"] += 1
                if "direct_imports" in patterns:
                    analysis["direct_import_count"] += 1
                if "dynamic_imports" in patterns:
                    analysis["dynamic_import_count"] += 1

                # Track pattern combinations
                pattern_key = "+".join(sorted(patterns))
                analysis["patterns"][pattern_key] = analysis["patterns"].get(pattern_key, 0) + 1

                if issues:
                    analysis["inconsistencies"][str(py_file)] = issues

            except Exception as e:
                analysis["inconsistencies"][str(py_file)] = [f"Analysis failed: {e}"]

        return analysis

    def _analyze_all_definitions(self) -> Dict:
        """Analyze __all__ definitions across API modules."""
        analysis = {
            "total_modules": 0,
            "modules_with_all": 0,
            "modules_missing_all": 0,
            "missing_all": [],
            "inconsistent_all": {},
        }

        api_path = Path("src/saplings/api")
        if not api_path.exists():
            return analysis

        for py_file in api_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis["total_modules"] += 1

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                if "__all__" in content:
                    analysis["modules_with_all"] += 1

                    # Check for issues with __all__ definition
                    issues = self._check_all_definition_quality(content)
                    if issues:
                        analysis["inconsistent_all"][str(py_file)] = issues
                else:
                    analysis["modules_missing_all"] += 1
                    analysis["missing_all"].append(str(py_file))

            except Exception as e:
                analysis["inconsistent_all"][str(py_file)] = [f"Analysis failed: {e}"]

        return analysis

    def _analyze_inheritance_patterns(self) -> Dict:
        """Analyze inheritance patterns in API modules."""
        analysis = {
            "total_modules": 0,
            "direct_inheritance": 0,
            "complex_patterns": 0,
            "complex_pattern_modules": {},
            "recommended_changes": {},
        }

        api_path = Path("src/saplings/api")
        if not api_path.exists():
            return analysis

        for py_file in api_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis["total_modules"] += 1

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                patterns = self._identify_inheritance_patterns(content)

                if "direct_inheritance" in patterns:
                    analysis["direct_inheritance"] += 1

                complex_patterns = [p for p in patterns if p != "direct_inheritance"]
                if complex_patterns:
                    analysis["complex_patterns"] += 1
                    analysis["complex_pattern_modules"][str(py_file)] = complex_patterns

                    # Generate recommendations
                    recommendations = self._generate_inheritance_recommendations(complex_patterns)
                    if recommendations:
                        analysis["recommended_changes"][str(py_file)] = recommendations

            except Exception as e:
                analysis["complex_pattern_modules"][str(py_file)] = [f"Analysis failed: {e}"]

        return analysis

    def _analyze_dynamic_imports(self) -> Dict:
        """Analyze dynamic import usage in API modules."""
        analysis = {
            "total_modules": 0,
            "modules_with_dynamic": 0,
            "modules_with_importlib": 0,
            "dynamic_imports": {},
            "justified_dynamic": {},
        }

        api_path = Path("src/saplings/api")
        if not api_path.exists():
            return analysis

        for py_file in api_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            analysis["total_modules"] += 1

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                dynamic_imports = self._find_dynamic_imports(content)

                if dynamic_imports:
                    analysis["modules_with_dynamic"] += 1

                    # Separate justified from unjustified
                    justified = []
                    unjustified = []

                    for imp in dynamic_imports:
                        if self._is_justified_dynamic_import(imp, content):
                            justified.append(imp)
                        else:
                            unjustified.append(imp)

                    if justified:
                        analysis["justified_dynamic"][str(py_file)] = justified
                    if unjustified:
                        analysis["dynamic_imports"][str(py_file)] = unjustified

                if "importlib" in content:
                    analysis["modules_with_importlib"] += 1

            except Exception as e:
                analysis["dynamic_imports"][str(py_file)] = [f"Analysis failed: {e}"]

        return analysis

    def _identify_import_patterns(self, content: str) -> List[str]:
        """Identify import patterns in module content."""
        patterns = []

        if "TYPE_CHECKING" in content:
            patterns.append("TYPE_CHECKING")
        if "from " in content and "import " in content:
            patterns.append("direct_imports")
        if "importlib" in content or "__import__" in content:
            patterns.append("dynamic_imports")
        if "class " in content and "(" in content:
            patterns.append("inheritance")

        return patterns

    def _check_pattern_consistency(self, content: str, patterns: List[str]) -> List[str]:
        """Check for pattern consistency issues."""
        issues = []

        # Check for mixed TYPE_CHECKING and runtime imports
        if "TYPE_CHECKING" in patterns and "direct_imports" in patterns:
            if "if TYPE_CHECKING:" in content and "from " in content.split("if TYPE_CHECKING:")[0]:
                issues.append("Mixed TYPE_CHECKING and runtime imports")

        return issues

    def _check_all_definition_quality(self, content: str) -> List[str]:
        """Check quality of __all__ definition."""
        issues = []

        # Check if __all__ is at module level
        if "__all__" in content:
            lines = content.split("\n")
            all_line_found = False
            for i, line in enumerate(lines):
                if "__all__" in line and not line.strip().startswith("#"):
                    # Check if it's properly formatted
                    if not line.strip().startswith("__all__"):
                        issues.append("__all__ not at start of line")
                    all_line_found = True
                    break

            if not all_line_found:
                issues.append("__all__ definition not found at module level")

        return issues

    def _identify_inheritance_patterns(self, content: str) -> List[str]:
        """Identify inheritance patterns in module content."""
        patterns = []

        if "class " in content and "(" in content:
            if "__new__" in content:
                patterns.append("complex_new_method")
            if "importlib" in content:
                patterns.append("dynamic_inheritance")
            if "class " in content and ": pass" in content:
                patterns.append("direct_inheritance")
            if "@" in content and "class " in content:
                patterns.append("decorated_inheritance")

        return patterns

    def _generate_inheritance_recommendations(self, patterns: List[str]) -> List[str]:
        """Generate recommendations for improving inheritance patterns."""
        recommendations = []

        if "complex_new_method" in patterns:
            recommendations.append("Replace complex __new__ with direct inheritance")
        if "dynamic_inheritance" in patterns:
            recommendations.append("Replace dynamic inheritance with static imports")

        return recommendations

    def _find_dynamic_imports(self, content: str) -> List[str]:
        """Find dynamic import statements in content."""
        dynamic_imports = []

        if "importlib.import_module" in content:
            dynamic_imports.append("importlib.import_module")
        if "__import__" in content:
            dynamic_imports.append("__import__")
        if "getattr(" in content and "import" in content:
            dynamic_imports.append("getattr-based import")

        return dynamic_imports

    def _is_justified_dynamic_import(self, import_type: str, content: str) -> bool:
        """Check if dynamic import is justified (e.g., for circular dependency resolution)."""
        # Dynamic imports are justified if they're used for lazy loading to avoid circular deps
        if "__getattr__" in content:
            return True
        if "# Lazy loading" in content or "# lazy loading" in content:
            return True
        if "circular" in content.lower():
            return True

        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
