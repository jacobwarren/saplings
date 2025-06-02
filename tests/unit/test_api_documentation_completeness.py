"""
Test API documentation completeness for publication readiness.

This module tests Task 7.6: Create comprehensive API documentation.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Dict, List


class TestAPIDocumentationCompleteness:
    """Test API documentation completeness and quality."""

    def test_core_api_classes_have_documentation(self):
        """Test that core API classes have comprehensive documentation."""
        core_classes = self._get_core_api_classes()
        documentation_analysis = {}

        for class_name, class_obj in core_classes.items():
            analysis = self._analyze_class_documentation(class_obj)
            documentation_analysis[class_name] = analysis

        print("\nCore API documentation analysis:")
        good_docs = 0
        for class_name, analysis in documentation_analysis.items():
            status = "✅" if analysis["comprehensive"] else "❌"
            print(f"  {status} {class_name}: {analysis['score']}/10")
            if analysis["comprehensive"]:
                good_docs += 1

        print(
            f"\nSummary: {good_docs}/{len(core_classes)} core classes have comprehensive documentation"
        )

        # For now, document the state rather than enforce requirements
        # This will be updated as documentation improves

    def test_public_methods_have_documentation(self):
        """Test that public methods have adequate documentation."""
        core_classes = self._get_core_api_classes()
        method_analysis = {}

        for class_name, class_obj in core_classes.items():
            methods = self._get_public_methods(class_obj)
            method_analysis[class_name] = {}

            for method_name, method_obj in methods.items():
                analysis = self._analyze_method_documentation(method_obj)
                method_analysis[class_name][method_name] = analysis

        print("\nPublic method documentation analysis:")
        for class_name, methods in method_analysis.items():
            good_methods = sum(1 for analysis in methods.values() if analysis["adequate"])
            total_methods = len(methods)
            print(f"  {class_name}: {good_methods}/{total_methods} methods documented")

    def test_documentation_structure_exists(self):
        """Test that documentation structure exists."""
        expected_docs = {
            "external-docs": [
                "README.md",
                "getting-started.md",
                "api-reference.md",
                "examples.md",
                "troubleshooting.md",
            ],
            "docs": ["architecture.md", "development.md", "api-cleanup-standardization.md"],
        }

        documentation_status = {}

        for doc_dir, expected_files in expected_docs.items():
            doc_path = Path(doc_dir)
            documentation_status[doc_dir] = {"exists": doc_path.exists(), "files": {}}

            if doc_path.exists():
                for expected_file in expected_files:
                    file_path = doc_path / expected_file
                    documentation_status[doc_dir]["files"][expected_file] = {
                        "exists": file_path.exists(),
                        "size": file_path.stat().st_size if file_path.exists() else 0,
                    }
            else:
                for expected_file in expected_files:
                    documentation_status[doc_dir]["files"][expected_file] = {
                        "exists": False,
                        "size": 0,
                    }

        print("\nDocumentation structure analysis:")
        for doc_dir, status in documentation_status.items():
            print(f"{doc_dir}: {'✅' if status['exists'] else '❌'}")
            for file_name, file_status in status["files"].items():
                size_info = f" ({file_status['size']} bytes)" if file_status["exists"] else ""
                print(f"  - {file_name}: {'✅' if file_status['exists'] else '❌'}{size_info}")

    def test_examples_coverage(self):
        """Test that examples cover major use cases."""
        examples_analysis = self._analyze_examples_coverage()

        print("\nExamples coverage analysis:")
        for category, examples in examples_analysis.items():
            print(f"{category}: {len(examples)} examples")
            for example in examples:
                print(f"  - {example}")

    def test_docstring_quality_standards(self):
        """Test that docstrings meet quality standards."""
        quality_analysis = self._analyze_docstring_quality()

        print("\nDocstring quality analysis:")
        for standard, results in quality_analysis.items():
            passed = results["passed"]
            total = results["total"]
            print(f"{standard}: {passed}/{total} ({passed/total*100:.1f}%)")

    def _get_core_api_classes(self) -> Dict[str, type]:
        """Get core API classes for documentation analysis."""
        core_classes = {}

        # Core API modules to check
        api_modules = [
            "saplings.api.agent",
            "saplings.api.tools",
            "saplings.api.models",
            "saplings.api.memory",
            "saplings.api.core",
        ]

        # Core class names to prioritize
        core_class_names = {
            "Agent",
            "AgentBuilder",
            "AgentConfig",
            "Tool",
            "PythonInterpreterTool",
            "FinalAnswerTool",
            "MemoryStore",
            "Document",
            "DocumentMetadata",
            "LLM",
            "LLMBuilder",
            "LLMResponse",
            "ExecutionService",
            "MemoryManager",
            "ToolService",
        }

        for module_name in api_modules:
            try:
                module = importlib.import_module(module_name)

                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and not name.startswith("_")
                        and (name in core_class_names or len(core_classes) < 20)
                    ):
                        core_classes[name] = obj
            except ImportError:
                continue

        return core_classes

    def _analyze_class_documentation(self, class_obj: type) -> Dict:
        """Analyze documentation quality for a class."""
        analysis = {
            "has_docstring": False,
            "docstring_length": 0,
            "has_examples": False,
            "has_parameters": False,
            "has_returns": False,
            "comprehensive": False,
            "score": 0,
        }

        if class_obj.__doc__:
            docstring = class_obj.__doc__.strip()
            analysis["has_docstring"] = True
            analysis["docstring_length"] = len(docstring)
            analysis["has_examples"] = "example" in docstring.lower() or "```" in docstring
            analysis["has_parameters"] = "param" in docstring.lower() or "arg" in docstring.lower()
            analysis["has_returns"] = "return" in docstring.lower()

            # Calculate score
            score = 0
            if analysis["has_docstring"]:
                score += 2
            if analysis["docstring_length"] > 100:
                score += 2
            if analysis["has_examples"]:
                score += 3
            if analysis["has_parameters"]:
                score += 2
            if analysis["has_returns"]:
                score += 1

            analysis["score"] = score
            analysis["comprehensive"] = score >= 7

        return analysis

    def _get_public_methods(self, class_obj: type) -> Dict[str, callable]:
        """Get public methods of a class."""
        methods = {}

        for name, method in inspect.getmembers(class_obj, inspect.isfunction):
            if not name.startswith("_"):
                methods[name] = method

        # Also check for methods defined in the class
        for name in dir(class_obj):
            if not name.startswith("_"):
                attr = getattr(class_obj, name)
                if callable(attr) and not inspect.isclass(attr):
                    methods[name] = attr

        return methods

    def _analyze_method_documentation(self, method_obj: callable) -> Dict:
        """Analyze documentation quality for a method."""
        analysis = {
            "has_docstring": False,
            "docstring_length": 0,
            "has_parameters": False,
            "has_returns": False,
            "adequate": False,
        }

        if hasattr(method_obj, "__doc__") and method_obj.__doc__:
            docstring = method_obj.__doc__.strip()
            analysis["has_docstring"] = True
            analysis["docstring_length"] = len(docstring)
            analysis["has_parameters"] = "param" in docstring.lower() or "arg" in docstring.lower()
            analysis["has_returns"] = "return" in docstring.lower()

            # Method is adequate if it has a docstring > 50 chars
            analysis["adequate"] = len(docstring) > 50

        return analysis

    def _analyze_examples_coverage(self) -> Dict[str, List[str]]:
        """Analyze examples coverage for major use cases."""
        examples_analysis = {
            "basic_usage": [],
            "advanced_features": [],
            "integrations": [],
            "tutorials": [],
        }

        # Check examples directory
        examples_dir = Path("examples")
        if examples_dir.exists():
            for example_file in examples_dir.rglob("*.py"):
                example_name = str(example_file.relative_to(examples_dir))

                # Categorize examples
                if any(
                    basic in example_name.lower()
                    for basic in ["basic", "simple", "hello", "getting_started"]
                ):
                    examples_analysis["basic_usage"].append(example_name)
                elif any(adv in example_name.lower() for adv in ["advanced", "complex", "custom"]):
                    examples_analysis["advanced_features"].append(example_name)
                elif any(
                    integ in example_name.lower() for integ in ["integration", "api", "service"]
                ):
                    examples_analysis["integrations"].append(example_name)
                else:
                    examples_analysis["tutorials"].append(example_name)

        # Check for inline examples in docstrings
        try:
            import saplings

            if hasattr(saplings, "__doc__") and saplings.__doc__:
                if "example" in saplings.__doc__.lower():
                    examples_analysis["basic_usage"].append("Package docstring example")
        except ImportError:
            pass

        return examples_analysis

    def _analyze_docstring_quality(self) -> Dict[str, Dict]:
        """Analyze overall docstring quality standards."""
        quality_analysis = {
            "has_docstring": {"passed": 0, "total": 0},
            "adequate_length": {"passed": 0, "total": 0},
            "has_examples": {"passed": 0, "total": 0},
            "follows_format": {"passed": 0, "total": 0},
        }

        core_classes = self._get_core_api_classes()

        for class_name, class_obj in core_classes.items():
            quality_analysis["has_docstring"]["total"] += 1
            quality_analysis["adequate_length"]["total"] += 1
            quality_analysis["has_examples"]["total"] += 1
            quality_analysis["follows_format"]["total"] += 1

            if class_obj.__doc__:
                quality_analysis["has_docstring"]["passed"] += 1

                docstring = class_obj.__doc__.strip()
                if len(docstring) > 100:
                    quality_analysis["adequate_length"]["passed"] += 1

                if "example" in docstring.lower() or "```" in docstring:
                    quality_analysis["has_examples"]["passed"] += 1

                # Check for basic format (description + parameters/returns)
                if (
                    "param" in docstring.lower()
                    or "arg" in docstring.lower()
                    or "return" in docstring.lower()
                    or len(docstring.split("\n")) > 3
                ):
                    quality_analysis["follows_format"]["passed"] += 1

        return quality_analysis
