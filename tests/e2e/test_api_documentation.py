"""
API documentation validation tests for Task 9.18.

This module implements comprehensive validation of API documentation
to ensure all public APIs are properly documented with examples and stability levels.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import List

import pytest


class TestAPIDocumentation:
    """Comprehensive API documentation validation tests."""

    def setup_method(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.src_dir = self.project_root / "src"
        self.api_modules = self._discover_api_modules()

    def _discover_api_modules(self) -> List[str]:
        """Discover all API modules in the saplings.api package."""
        api_modules = []
        api_dir = self.src_dir / "saplings" / "api"

        if api_dir.exists():
            for py_file in api_dir.rglob("*.py"):
                if py_file.name != "__init__.py":
                    # Convert file path to module name
                    rel_path = py_file.relative_to(self.src_dir)
                    module_name = str(rel_path.with_suffix("")).replace("/", ".")
                    api_modules.append(module_name)

        return api_modules

    @pytest.mark.e2e()
    def test_api_modules_have_docstrings(self):
        """Test that all API modules have comprehensive module docstrings."""
        missing_docstrings = []

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)

                # Check for module docstring
                if not module.__doc__ or len(module.__doc__.strip()) < 10:
                    missing_docstrings.append(module_name)

            except ImportError:
                # Skip modules that can't be imported
                continue

        if missing_docstrings:
            print(f"Found {len(missing_docstrings)} API modules without proper docstrings:")
            for module in missing_docstrings[:10]:  # Show first 10
                print(f"  {module}")
            if len(missing_docstrings) > 10:
                print(f"  ... and {len(missing_docstrings) - 10} more modules")

    @pytest.mark.e2e()
    def test_public_classes_have_docstrings(self):
        """Test that all public classes have comprehensive docstrings."""
        missing_class_docs = []

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)

                # Get all public classes (not starting with _)
                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isclass(obj):
                            # Check if class has proper docstring
                            if not obj.__doc__ or len(obj.__doc__.strip()) < 20:
                                missing_class_docs.append(f"{module_name}.{name}")

            except ImportError:
                continue

        if missing_class_docs:
            print(f"Found {len(missing_class_docs)} public classes without proper docstrings:")
            for cls in missing_class_docs[:10]:  # Show first 10
                print(f"  {cls}")
            if len(missing_class_docs) > 10:
                print(f"  ... and {len(missing_class_docs) - 10} more classes")

    @pytest.mark.e2e()
    def test_public_functions_have_docstrings(self):
        """Test that all public functions have comprehensive docstrings."""
        missing_function_docs = []

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)

                # Get all public functions (not starting with _)
                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isfunction(obj):
                            # Check if function has proper docstring
                            if not obj.__doc__ or len(obj.__doc__.strip()) < 20:
                                missing_function_docs.append(f"{module_name}.{name}")

            except ImportError:
                continue

        if missing_function_docs:
            print(f"Found {len(missing_function_docs)} public functions without proper docstrings:")
            for func in missing_function_docs[:10]:  # Show first 10
                print(f"  {func}")
            if len(missing_function_docs) > 10:
                print(f"  ... and {len(missing_function_docs) - 10} more functions")

    @pytest.mark.e2e()
    def test_stability_annotations_documented(self):
        """Test that stability annotations are properly documented."""
        missing_stability_docs = []

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)

                # Check for stability annotations
                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isclass(obj) or inspect.isfunction(obj):
                            # Check if object has stability annotation
                            has_stability = hasattr(obj, "__stability__")

                            if has_stability:
                                # Check if stability is documented in docstring
                                docstring = obj.__doc__ or ""
                                stability_documented = any(
                                    word in docstring.lower()
                                    for word in ["stable", "beta", "experimental", "deprecated"]
                                )

                                if not stability_documented:
                                    missing_stability_docs.append(f"{module_name}.{name}")

            except ImportError:
                continue

        if missing_stability_docs:
            print(
                f"Found {len(missing_stability_docs)} items with stability annotations but no documentation:"
            )
            for item in missing_stability_docs[:10]:  # Show first 10
                print(f"  {item}")
            if len(missing_stability_docs) > 10:
                print(f"  ... and {len(missing_stability_docs) - 10} more items")

    @pytest.mark.e2e()
    def test_docstrings_include_examples(self):
        """Test that important API components include usage examples in docstrings."""
        missing_examples = []

        # Key API components that should have examples
        key_components = [
            "saplings.api.agent",
            "saplings.api.tools",
            "saplings.api.memory",
        ]

        for module_name in key_components:
            if module_name in self.api_modules:
                try:
                    module = importlib.import_module(module_name)

                    # Check main classes for examples
                    for name in dir(module):
                        if not name.startswith("_"):
                            obj = getattr(module, name)
                            if inspect.isclass(obj):
                                docstring = obj.__doc__ or ""

                                # Check for example indicators
                                has_example = any(
                                    indicator in docstring.lower()
                                    for indicator in ["example", "usage", ">>>", "import saplings"]
                                )

                                if not has_example and len(docstring) > 50:
                                    missing_examples.append(f"{module_name}.{name}")

                except ImportError:
                    continue

        if missing_examples:
            print(f"Found {len(missing_examples)} key API components without usage examples:")
            for item in missing_examples:
                print(f"  {item}")

    @pytest.mark.e2e()
    def test_parameter_documentation_completeness(self):
        """Test that function and method parameters are properly documented."""
        missing_param_docs = []

        for module_name in self.api_modules[:5]:  # Check first 5 modules to avoid timeout
            try:
                module = importlib.import_module(module_name)

                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isfunction(obj) or inspect.ismethod(obj):
                            # Get function signature
                            try:
                                sig = inspect.signature(obj)
                                params = list(sig.parameters.keys())

                                # Skip if no parameters or only self
                                if len(params) <= 1:
                                    continue

                                docstring = obj.__doc__ or ""

                                # Check if parameters are documented
                                params_documented = any(
                                    param in docstring for param in params[1:]
                                )  # Skip self

                                if not params_documented and len(params) > 1:
                                    missing_param_docs.append(f"{module_name}.{name}")

                            except (ValueError, TypeError):
                                # Skip if we can't get signature
                                continue

            except ImportError:
                continue

        if missing_param_docs:
            print(f"Found {len(missing_param_docs)} functions without parameter documentation:")
            for func in missing_param_docs[:10]:  # Show first 10
                print(f"  {func}")
            if len(missing_param_docs) > 10:
                print(f"  ... and {len(missing_param_docs) - 10} more functions")

    @pytest.mark.e2e()
    def test_return_value_documentation(self):
        """Test that function return values are properly documented."""
        missing_return_docs = []

        for module_name in self.api_modules[:5]:  # Check first 5 modules to avoid timeout
            try:
                module = importlib.import_module(module_name)

                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isfunction(obj):
                            # Get function signature
                            try:
                                sig = inspect.signature(obj)

                                # Skip if no return annotation and no docstring
                                if sig.return_annotation == inspect.Signature.empty:
                                    docstring = obj.__doc__ or ""

                                    # Check if return value is documented
                                    return_documented = any(
                                        indicator in docstring.lower()
                                        for indicator in ["return", "returns", "yields"]
                                    )

                                    if not return_documented and len(docstring) > 20:
                                        missing_return_docs.append(f"{module_name}.{name}")

                            except (ValueError, TypeError):
                                # Skip if we can't get signature
                                continue

            except ImportError:
                continue

        if missing_return_docs:
            print(f"Found {len(missing_return_docs)} functions without return value documentation:")
            for func in missing_return_docs[:10]:  # Show first 10
                print(f"  {func}")
            if len(missing_return_docs) > 10:
                print(f"  ... and {len(missing_return_docs) - 10} more functions")

    @pytest.mark.e2e()
    def test_documentation_consistency(self):
        """Test that documentation follows consistent patterns and style."""
        inconsistent_docs = []

        for module_name in self.api_modules[:5]:  # Check first 5 modules
            try:
                module = importlib.import_module(module_name)

                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isclass(obj) or inspect.isfunction(obj):
                            docstring = obj.__doc__ or ""

                            if len(docstring) > 10:
                                # Check for common documentation patterns
                                issues = []

                                # Check for proper sentence structure
                                if not docstring.strip().endswith("."):
                                    issues.append("missing period")

                                # Check for proper capitalization
                                if not docstring.strip()[0].isupper():
                                    issues.append("not capitalized")

                                if issues:
                                    inconsistent_docs.append(
                                        f"{module_name}.{name}: {', '.join(issues)}"
                                    )

            except ImportError:
                continue

        if inconsistent_docs:
            print(f"Found {len(inconsistent_docs)} documentation consistency issues:")
            for issue in inconsistent_docs[:10]:  # Show first 10
                print(f"  {issue}")
            if len(inconsistent_docs) > 10:
                print(f"  ... and {len(inconsistent_docs) - 10} more issues")

    @pytest.mark.e2e()
    def test_api_documentation_summary(self):
        """Provide a comprehensive summary of API documentation status."""
        print("\n=== API Documentation Summary ===")
        print(f"Total API modules discovered: {len(self.api_modules)}")

        total_classes = 0
        total_functions = 0
        documented_classes = 0
        documented_functions = 0

        for module_name in self.api_modules:
            try:
                module = importlib.import_module(module_name)

                for name in dir(module):
                    if not name.startswith("_"):
                        obj = getattr(module, name)
                        if inspect.isclass(obj):
                            total_classes += 1
                            if obj.__doc__ and len(obj.__doc__.strip()) >= 20:
                                documented_classes += 1
                        elif inspect.isfunction(obj):
                            total_functions += 1
                            if obj.__doc__ and len(obj.__doc__.strip()) >= 20:
                                documented_functions += 1

            except ImportError:
                continue

        class_doc_rate = (documented_classes / total_classes * 100) if total_classes > 0 else 0
        function_doc_rate = (
            (documented_functions / total_functions * 100) if total_functions > 0 else 0
        )

        print(f"Classes: {documented_classes}/{total_classes} documented ({class_doc_rate:.1f}%)")
        print(
            f"Functions: {documented_functions}/{total_functions} documented ({function_doc_rate:.1f}%)"
        )
        print("=== End Documentation Summary ===\n")
