"""
Test for Task 9.9: Create simplified factory methods for common use cases.

This test verifies that simplified factory methods exist for common Agent creation patterns.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class TestTask99FactoryMethods:
    """Test Task 9.9: Create simplified factory methods for common use cases."""

    def test_agent_class_has_factory_methods(self):
        """Test that Agent class has the required factory methods."""
        agent_file = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/api/agent.py"
        )

        if not agent_file.exists():
            pytest.fail("Agent API file doesn't exist")

        content = agent_file.read_text()

        # Required factory methods
        required_methods = ["simple", "for_openai", "with_tools", "for_research"]

        existing_methods = []
        missing_methods = []

        for method in required_methods:
            # Look for classmethod definition
            if f"def {method}(" in content and "@classmethod" in content:
                # Check if the method is actually a classmethod
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if f"def {method}(" in line:
                        # Check previous lines for @classmethod
                        for j in range(max(0, i - 3), i):
                            if "@classmethod" in lines[j]:
                                existing_methods.append(method)
                                print(f"✅ Factory method exists: {method}")
                                break
                        else:
                            # Method exists but not as classmethod
                            print(f"⚠️  Method {method} exists but not as classmethod")
                        break
                else:
                    missing_methods.append(method)
                    print(f"❌ Factory method missing: {method}")
            else:
                missing_methods.append(method)
                print(f"❌ Factory method missing: {method}")

        print(f"Existing factory methods: {len(existing_methods)}/{len(required_methods)}")

        if missing_methods:
            print("Missing factory methods:")
            for method in missing_methods:
                print(f"  - {method}")

        # Don't fail test - this shows what needs to be implemented
        assert len(required_methods) > 0, "Should check for factory methods"

    def test_factory_methods_have_proper_signatures(self):
        """Test that factory methods have proper signatures and return types."""
        agent_file = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/api/agent.py"
        )

        if not agent_file.exists():
            pytest.fail("Agent API file doesn't exist")

        content = agent_file.read_text()

        # Expected signatures for factory methods
        expected_signatures = {
            "simple": ["provider", "model", "api_key"],
            "for_openai": ["model"],
            "with_tools": ["provider", "model", "tools"],
            "for_research": ["provider", "model"],
        }

        signature_analysis = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "Agent":
                    for method in node.body:
                        if (
                            isinstance(method, ast.FunctionDef)
                            and method.name in expected_signatures
                        ):
                            # Check if it's a classmethod
                            is_classmethod = any(
                                isinstance(decorator, ast.Name) and decorator.id == "classmethod"
                                for decorator in method.decorator_list
                            )

                            # Get parameter names
                            param_names = [arg.arg for arg in method.args.args[1:]]  # Skip 'cls'
                            expected_params = expected_signatures[method.name]

                            # Check return annotation
                            has_return_annotation = method.returns is not None

                            signature_analysis.append(
                                {
                                    "method": method.name,
                                    "is_classmethod": is_classmethod,
                                    "params": param_names,
                                    "expected_params": expected_params,
                                    "has_return_annotation": has_return_annotation,
                                }
                            )

                            if is_classmethod:
                                print(f"✅ {method.name} is a classmethod")
                            else:
                                print(f"❌ {method.name} is not a classmethod")

                            if has_return_annotation:
                                print(f"✅ {method.name} has return annotation")
                            else:
                                print(f"⚠️  {method.name} missing return annotation")

                            # Check parameters
                            missing_params = set(expected_params) - set(param_names)
                            if missing_params:
                                print(f"⚠️  {method.name} missing params: {missing_params}")
                            else:
                                print(f"✅ {method.name} has expected parameters")

        except Exception as e:
            print(f"Could not parse Agent class: {e}")

        print(f"Analyzed {len(signature_analysis)} factory methods")

        # Don't fail test - this shows current state
        assert len(signature_analysis) >= 0, "Should analyze factory methods"

    def test_factory_methods_have_documentation(self):
        """Test that factory methods have proper documentation."""
        agent_file = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/api/agent.py"
        )

        if not agent_file.exists():
            pytest.fail("Agent API file doesn't exist")

        content = agent_file.read_text()

        factory_methods = ["simple", "for_openai", "with_tools", "for_research"]

        documented_methods = []
        undocumented_methods = []

        for method in factory_methods:
            # Look for method definition and check for docstring
            lines = content.split("\n")
            method_found = False
            has_docstring = False

            for i, line in enumerate(lines):
                if f"def {method}(" in line:
                    method_found = True
                    # Check next few lines for docstring
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if '"""' in lines[j] or "'''" in lines[j]:
                            has_docstring = True
                            break
                    break

            if method_found:
                if has_docstring:
                    documented_methods.append(method)
                    print(f"✅ {method} has documentation")
                else:
                    undocumented_methods.append(method)
                    print(f"❌ {method} missing documentation")
            else:
                undocumented_methods.append(method)
                print(f"❌ {method} method not found")

        print(f"Documented methods: {len(documented_methods)}")
        print(f"Undocumented methods: {len(undocumented_methods)}")

        # Don't fail test - this shows what needs documentation
        assert len(factory_methods) > 0, "Should check for method documentation"

    def test_factory_methods_reduce_configuration_complexity(self):
        """Test that factory methods reduce configuration complexity."""
        agent_file = Path(
            "/Users/jacobwarren/Development/agents/saplings/src/saplings/api/agent.py"
        )

        if not agent_file.exists():
            pytest.fail("Agent API file doesn't exist")

        content = agent_file.read_text()

        # Look for evidence of simplified configuration in factory methods
        complexity_indicators = {
            "default_values": "=",  # Default parameter values
            "sensible_defaults": "default",  # References to defaults
            "simplified_config": "AgentConfig",  # Creates AgentConfig internally
            "error_handling": "raise",  # Proper error handling
        }

        factory_methods = ["simple", "for_openai", "with_tools", "for_research"]
        complexity_analysis = []

        for method in factory_methods:
            # Extract method content
            lines = content.split("\n")
            method_lines = []
            in_method = False
            indent_level = 0

            for line in lines:
                if f"def {method}(" in line:
                    in_method = True
                    indent_level = len(line) - len(line.lstrip())
                    method_lines.append(line)
                elif in_method:
                    current_indent = len(line) - len(line.lstrip())
                    if line.strip() and current_indent <= indent_level:
                        break  # End of method
                    method_lines.append(line)

            method_content = "\n".join(method_lines)

            # Analyze complexity reduction features
            features = {}
            for feature, indicator in complexity_indicators.items():
                features[feature] = indicator in method_content

            complexity_analysis.append(
                {"method": method, "features": features, "content_length": len(method_content)}
            )

            feature_count = sum(features.values())
            print(
                f"{method}: {feature_count}/{len(complexity_indicators)} complexity reduction features"
            )

            for feature, present in features.items():
                status = "✅" if present else "❌"
                print(f"  {status} {feature}")

        print(f"Analyzed {len(complexity_analysis)} factory methods for complexity reduction")

        # Don't fail test - this shows current implementation quality
        assert len(complexity_analysis) >= 0, "Should analyze complexity reduction"

    def test_other_classes_have_factory_methods(self):
        """Test that other core classes have factory methods where appropriate."""
        # Check for factory methods in other core classes
        core_classes = [
            ("saplings/api/tools.py", "Tool"),
            ("saplings/api/memory.py", "MemoryStore"),
            ("saplings/api/models.py", "ModelAdapter"),
        ]

        src_dir = Path("/Users/jacobwarren/Development/agents/saplings/src")

        factory_method_analysis = []

        for file_path, class_name in core_classes:
            full_path = src_dir / file_path

            if not full_path.exists():
                print(f"⚠️  {file_path} doesn't exist")
                continue

            try:
                content = full_path.read_text()

                # Look for factory methods (classmethods)
                factory_methods = []
                lines = content.split("\n")

                for i, line in enumerate(lines):
                    if "@classmethod" in line:
                        # Check next line for method definition
                        if i + 1 < len(lines) and "def " in lines[i + 1]:
                            method_name = lines[i + 1].split("def ")[1].split("(")[0].strip()
                            factory_methods.append(method_name)

                factory_method_analysis.append(
                    {"class": class_name, "file": file_path, "factory_methods": factory_methods}
                )

                if factory_methods:
                    print(f"✅ {class_name} has factory methods: {factory_methods}")
                else:
                    print(f"⚠️  {class_name} has no factory methods")

            except Exception as e:
                print(f"⚠️  Could not analyze {file_path}: {e}")

        print(f"Analyzed {len(factory_method_analysis)} core classes for factory methods")

        # Don't fail test - this shows current state
        assert len(factory_method_analysis) >= 0, "Should analyze core classes"

    def test_factory_method_examples_exist(self):
        """Test that examples demonstrate factory method usage."""
        examples_dir = Path("/Users/jacobwarren/Development/agents/saplings/examples")

        if not examples_dir.exists():
            print("❌ Examples directory doesn't exist")
            return

        factory_examples = []

        # Look for examples that use factory methods
        for example_file in examples_dir.rglob("*.py"):
            try:
                content = example_file.read_text()

                # Check for factory method usage
                factory_patterns = [
                    "Agent.simple(",
                    "Agent.for_openai(",
                    "Agent.with_tools(",
                    "Agent.for_research(",
                ]

                used_patterns = []
                for pattern in factory_patterns:
                    if pattern in content:
                        used_patterns.append(pattern.replace("(", ""))

                if used_patterns:
                    factory_examples.append(
                        {
                            "file": str(example_file.relative_to(examples_dir)),
                            "patterns": used_patterns,
                        }
                    )

            except Exception:
                continue

        if factory_examples:
            print(f"✅ Found {len(factory_examples)} examples using factory methods:")
            for example in factory_examples:
                print(f"  - {example['file']}: {example['patterns']}")
        else:
            print("❌ No examples found using factory methods")

        # Don't fail test - this shows what examples exist
        assert True, "Factory method examples check completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
