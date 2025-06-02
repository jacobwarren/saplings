"""
Test import performance optimization for publication readiness.

This module tests Task 7.5: Optimize import performance.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Dict, List


class TestImportPerformanceOptimization:
    """Test import performance optimization and analysis."""

    def test_main_package_import_time(self):
        """Test that main package import completes within acceptable time."""
        # Clear any cached imports
        modules_to_clear = [name for name in sys.modules.keys() if name.startswith("saplings")]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Measure import time
        start_time = time.time()
        import saplings  # noqa: F401

        import_time = time.time() - start_time

        print(f"\nMain package import time: {import_time:.3f} seconds")

        # Target: < 2 seconds for main package import
        # This is a reasonable target for a complex package
        assert import_time < 2.0, f"Main package import took {import_time:.3f}s, should be < 2.0s"

    def test_submodule_import_times(self):
        """Test import times for major submodules."""
        submodules = [
            "saplings.api.agent",
            "saplings.api.tools",
            "saplings.api.models",
            "saplings.api.memory",
            "saplings.api.services",
        ]

        import_times = {}

        for module_name in submodules:
            # Clear module cache
            if module_name in sys.modules:
                del sys.modules[module_name]

            start_time = time.time()
            try:
                importlib.import_module(module_name)
                import_time = time.time() - start_time
                import_times[module_name] = import_time
            except ImportError as e:
                import_times[module_name] = f"Failed: {e}"

        print("\nSubmodule import times:")
        for module, time_or_error in import_times.items():
            if isinstance(time_or_error, float):
                print(f"  {module}: {time_or_error:.3f}s")
            else:
                print(f"  {module}: {time_or_error}")

        # Check that successful imports are reasonably fast
        for module, time_or_error in import_times.items():
            if isinstance(time_or_error, float):
                assert (
                    time_or_error < 1.0
                ), f"{module} import took {time_or_error:.3f}s, should be < 1.0s"

    def test_identify_heavy_imports(self):
        """Test identification of heavy imports that could benefit from lazy loading."""
        heavy_imports = self._analyze_heavy_imports()

        print("\nHeavy import analysis:")
        for category, imports in heavy_imports.items():
            print(f"{category}: {len(imports)} imports")
            for imp in imports[:3]:  # Show first 3 examples
                print(f"  - {imp}")

    def test_lazy_loading_opportunities(self):
        """Test identification of lazy loading opportunities."""
        opportunities = self._identify_lazy_loading_opportunities()

        print("\nLazy loading opportunities:")
        for opportunity_type, modules in opportunities.items():
            print(f"{opportunity_type}: {len(modules)} modules")
            for module in modules[:3]:  # Show first 3 examples
                print(f"  - {module}")

    def test_import_dependency_analysis(self):
        """Test analysis of import dependencies to identify optimization targets."""
        dependency_analysis = self._analyze_import_dependencies()

        print("\nImport dependency analysis:")
        print(f"Total modules analyzed: {dependency_analysis['total_modules']}")
        print(
            f"Modules with heavy dependencies: {len(dependency_analysis['heavy_dependency_modules'])}"
        )
        print(f"Circular dependency candidates: {len(dependency_analysis['circular_candidates'])}")
        print(f"Optimization targets: {len(dependency_analysis['optimization_targets'])}")

        # Show top optimization targets
        print("\nTop optimization targets:")
        for target in dependency_analysis["optimization_targets"][:5]:
            print(f"  - {target['module']}: {target['reason']}")

    def test_current_lazy_loading_effectiveness(self):
        """Test effectiveness of current lazy loading patterns."""
        effectiveness = self._analyze_lazy_loading_effectiveness()

        print("\nLazy loading effectiveness:")
        for module, analysis in effectiveness.items():
            print(f"{module}:")
            print(f"  - Uses lazy loading: {analysis['uses_lazy_loading']}")
            print(f"  - Import time benefit: {analysis['estimated_benefit']}")
            print(f"  - Complexity cost: {analysis['complexity_cost']}")

    def _analyze_heavy_imports(self) -> Dict[str, List[str]]:
        """Analyze heavy imports that could benefit from optimization."""
        heavy_imports = {
            "ml_libraries": [],
            "visualization": [],
            "optional_tools": [],
            "large_frameworks": [],
        }

        src_path = Path("src/saplings")

        # Known heavy imports
        ml_patterns = ["torch", "transformers", "vllm", "triton", "accelerate"]
        viz_patterns = ["matplotlib", "plotly", "seaborn"]
        tool_patterns = ["selenium", "playwright", "requests", "beautifulsoup"]
        framework_patterns = ["opentelemetry", "langsmith", "anthropic", "openai"]

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                for pattern in ml_patterns:
                    if f"import {pattern}" in content:
                        heavy_imports["ml_libraries"].append(f"{py_file}: {pattern}")

                for pattern in viz_patterns:
                    if f"import {pattern}" in content:
                        heavy_imports["visualization"].append(f"{py_file}: {pattern}")

                for pattern in tool_patterns:
                    if f"import {pattern}" in content:
                        heavy_imports["optional_tools"].append(f"{py_file}: {pattern}")

                for pattern in framework_patterns:
                    if f"import {pattern}" in content:
                        heavy_imports["large_frameworks"].append(f"{py_file}: {pattern}")

            except Exception:
                continue

        return heavy_imports

    def _identify_lazy_loading_opportunities(self) -> Dict[str, List[str]]:
        """Identify opportunities for lazy loading."""
        opportunities = {
            "heavy_ml_imports": [],
            "optional_dependencies": [],
            "visualization_tools": [],
            "complex_builders": [],
        }

        src_path = Path("src/saplings")

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for heavy ML imports that aren't already lazy
                if any(lib in content for lib in ["torch", "transformers", "vllm"]):
                    if "try:" not in content or "__getattr__" not in content:
                        opportunities["heavy_ml_imports"].append(str(py_file))

                # Check for optional dependencies
                if any(dep in content for dep in ["selenium", "playwright", "langsmith"]):
                    if "try:" not in content:
                        opportunities["optional_dependencies"].append(str(py_file))

                # Check for visualization imports
                if any(viz in content for viz in ["matplotlib", "plotly"]):
                    opportunities["visualization_tools"].append(str(py_file))

                # Check for complex builders
                if "Builder" in str(py_file) and "import" in content:
                    opportunities["complex_builders"].append(str(py_file))

            except Exception:
                continue

        return opportunities

    def _analyze_import_dependencies(self) -> Dict:
        """Analyze import dependencies for optimization targets."""
        analysis = {
            "total_modules": 0,
            "heavy_dependency_modules": [],
            "circular_candidates": [],
            "optimization_targets": [],
        }

        src_path = Path("src/saplings")

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                analysis["total_modules"] += 1

                # Count imports
                import_count = content.count("import ")
                from_count = content.count("from ")
                total_imports = import_count + from_count

                # Heavy dependency modules
                if total_imports > 20:
                    analysis["heavy_dependency_modules"].append(
                        {"module": str(py_file), "import_count": total_imports}
                    )

                # Potential circular dependency candidates
                if "__getattr__" in content and "import" in content:
                    analysis["circular_candidates"].append(str(py_file))

                # Optimization targets
                if total_imports > 15 or any(
                    heavy in content for heavy in ["torch", "transformers", "matplotlib"]
                ):
                    reason = []
                    if total_imports > 15:
                        reason.append(f"{total_imports} imports")
                    if any(heavy in content for heavy in ["torch", "transformers"]):
                        reason.append("heavy ML libraries")
                    if "matplotlib" in content:
                        reason.append("visualization libraries")

                    analysis["optimization_targets"].append(
                        {"module": str(py_file), "reason": ", ".join(reason)}
                    )

            except Exception:
                continue

        # Sort optimization targets by potential impact
        analysis["optimization_targets"].sort(key=lambda x: len(x["reason"]), reverse=True)

        return analysis

    def _analyze_lazy_loading_effectiveness(self) -> Dict[str, Dict]:
        """Analyze effectiveness of current lazy loading patterns."""
        effectiveness = {}
        src_path = Path("src/saplings")

        # Find modules that use lazy loading
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                if "__getattr__" in content:
                    module_name = (
                        str(py_file.relative_to(src_path.parent))
                        .replace("/", ".")
                        .replace(".py", "")
                    )

                    # Analyze the lazy loading pattern
                    uses_lazy = "__getattr__" in content
                    has_heavy_imports = any(
                        heavy in content for heavy in ["torch", "transformers", "matplotlib"]
                    )

                    # Estimate benefit
                    if has_heavy_imports:
                        estimated_benefit = "High - defers heavy imports"
                    elif "import" in content:
                        estimated_benefit = "Medium - defers some imports"
                    else:
                        estimated_benefit = "Low - minimal import overhead"

                    # Estimate complexity cost
                    if content.count("__getattr__") > 1:
                        complexity_cost = "High - complex lazy loading"
                    elif "importlib" in content:
                        complexity_cost = "Medium - dynamic imports"
                    else:
                        complexity_cost = "Low - simple lazy loading"

                    effectiveness[module_name] = {
                        "uses_lazy_loading": uses_lazy,
                        "estimated_benefit": estimated_benefit,
                        "complexity_cost": complexity_cost,
                    }

            except Exception:
                continue

        return effectiveness
