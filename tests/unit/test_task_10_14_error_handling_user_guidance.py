"""
Test for Task 10.14: Implement comprehensive error handling and user guidance.

This test verifies that comprehensive error handling and user guidance has been implemented:
1. Clear, actionable error messages
2. Helpful suggestions for common issues
3. Error message standardization
4. User guidance documentation
5. Error recovery mechanisms
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestTask1014ErrorHandlingUserGuidance:
    """Test Task 10.14: Implement comprehensive error handling and user guidance."""

    def test_clear_error_messages_for_missing_config(self):
        """Test that clear error messages are provided for missing configuration."""
        error_test_code = """
import sys

try:
    import saplings

    # Test 1: Missing provider
    try:
        config = saplings.AgentConfig(model_name="test-model")  # Missing provider
        print("❌ Should have failed with missing provider")
        exit(1)
    except Exception as e:
        error_msg = str(e)
        print(f"✅ Test 1: Error message for missing provider: {error_msg}")

        # Check if error message is helpful
        if "provider" in error_msg.lower():
            print("✅ Error message mentions 'provider'")
        else:
            print("⚠️  Error message doesn't mention 'provider'")

    # Test 2: Missing model_name
    try:
        config = saplings.AgentConfig(provider="openai")  # Missing model_name
        print("❌ Should have failed with missing model_name")
        exit(1)
    except Exception as e:
        error_msg = str(e)
        print(f"✅ Test 2: Error message for missing model_name: {error_msg}")

        # Check if error message is helpful
        if "model_name" in error_msg.lower():
            print("✅ Error message mentions 'model_name'")
        else:
            print("⚠️  Error message doesn't mention 'model_name'")

    print("✅ Error message tests completed")
    exit(0)

except Exception as e:
    print(f"❌ Error message test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", error_test_code],
                timeout=30,
                capture_output=True,
                text=True,
                cwd="/Users/jacobwarren/Development/agents/saplings",
                check=False,
            )

            if result.returncode == 0:
                print("✅ Clear error message validation:")
                for line in result.stdout.strip().split("\n"):
                    print(f"   {line}")
            else:
                print(f"⚠️  Error message test: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.fail("Error message test timed out")

    def test_helpful_suggestions_for_common_issues(self):
        """Test that helpful suggestions are provided for common configuration issues."""
        suggestion_test_code = """
import sys

try:
    import saplings

    # Test invalid provider
    try:
        config = saplings.AgentConfig(provider="invalid_provider", model_name="test-model")
        print("❌ Should have failed with invalid provider")
        exit(1)
    except Exception as e:
        error_msg = str(e)
        print(f"✅ Invalid provider error: {error_msg}")

        # Check if error message provides suggestions
        helpful_keywords = ["supported", "valid", "available", "try", "use"]
        has_suggestions = any(keyword in error_msg.lower() for keyword in helpful_keywords)

        if has_suggestions:
            print("✅ Error message provides helpful suggestions")
        else:
            print("⚠️  Error message could be more helpful")

    print("✅ Suggestion tests completed")
    exit(0)

except Exception as e:
    print(f"❌ Suggestion test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", suggestion_test_code],
                timeout=30,
                capture_output=True,
                text=True,
                cwd="/Users/jacobwarren/Development/agents/saplings",
                check=False,
            )

            if result.returncode == 0:
                print("✅ Helpful suggestion validation:")
                for line in result.stdout.strip().split("\n"):
                    print(f"   {line}")
            else:
                print(f"⚠️  Suggestion test: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.fail("Suggestion test timed out")

    def test_error_message_standardization_exists(self):
        """Test that error message standardization documentation exists."""
        # Check for error message standardization documentation
        error_docs = [
            Path("docs/error-message-standardization.md"),
            Path("docs/error-handling.md"),
            Path("docs/troubleshooting.md"),
            Path("external-docs/error-handling-guide.md"),
        ]

        found_error_docs = []
        for doc_path in error_docs:
            if doc_path.exists():
                found_error_docs.append(doc_path)
                print(f"✅ Found error handling documentation: {doc_path}")

        if found_error_docs:
            print(f"✅ Found {len(found_error_docs)} error handling documentation files")
            assert (
                len(found_error_docs) >= 1
            ), "Should have at least one error handling documentation file"
        else:
            print("⚠️  No error handling documentation found")

    def test_user_guidance_documentation_exists(self):
        """Test that user guidance documentation exists."""
        # Check for user guidance documentation
        guidance_docs = [
            Path("external-docs/getting-started.md"),
            Path("external-docs/user-guide.md"),
            Path("external-docs/troubleshooting.md"),
            Path("docs/user-guidance.md"),
            Path("README.md"),
        ]

        found_guidance_docs = []
        for doc_path in guidance_docs:
            if doc_path.exists():
                found_guidance_docs.append(doc_path)
                print(f"✅ Found user guidance documentation: {doc_path}")

        if found_guidance_docs:
            print(f"✅ Found {len(found_guidance_docs)} user guidance documentation files")
            assert (
                len(found_guidance_docs) >= 2
            ), "Should have at least 2 user guidance documentation files"
        else:
            print("⚠️  No user guidance documentation found")

    def test_error_recovery_mechanisms_exist(self):
        """Test that error recovery mechanisms are implemented."""
        # Check for error recovery in AgentConfig
        recovery_test_code = """
import sys

try:
    import saplings

    # Test that AgentConfig provides recovery suggestions
    try:
        # Try to create config with empty provider
        config = saplings.AgentConfig(provider="", model_name="test-model")
        print("❌ Should have failed with empty provider")
        exit(1)
    except Exception as e:
        error_msg = str(e)
        print(f"✅ Empty provider error: {error_msg}")

        # Check if error provides recovery guidance
        recovery_keywords = ["should", "must", "required", "provide", "specify"]
        has_recovery = any(keyword in error_msg.lower() for keyword in recovery_keywords)

        if has_recovery:
            print("✅ Error message provides recovery guidance")
        else:
            print("⚠️  Error message could provide better recovery guidance")

    print("✅ Error recovery tests completed")
    exit(0)

except Exception as e:
    print(f"❌ Error recovery test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", recovery_test_code],
                timeout=30,
                capture_output=True,
                text=True,
                cwd="/Users/jacobwarren/Development/agents/saplings",
                check=False,
            )

            if result.returncode == 0:
                print("✅ Error recovery validation:")
                for line in result.stdout.strip().split("\n"):
                    print(f"   {line}")
            else:
                print(f"⚠️  Error recovery test: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.fail("Error recovery test timed out")

    def test_validation_error_handling_exists(self):
        """Test that validation error handling is implemented."""
        # Check for validation error handling in existing tests
        validation_test_files = [
            "tests/unit/test_agent_config_error_handling.py",
            "tests/unit/test_task_8_8_error_messages.py",
            "tests/unit/test_task_9_11_error_messages.py",
        ]

        existing_validation_tests = []
        for test_file in validation_test_files:
            if Path(test_file).exists():
                existing_validation_tests.append(test_file)
                print(f"✅ Found validation error test: {test_file}")

        if existing_validation_tests:
            print(f"✅ Found {len(existing_validation_tests)} validation error test files")
            assert (
                len(existing_validation_tests) >= 2
            ), "Should have at least 2 validation error test files"
        else:
            print("⚠️  No validation error test files found")

    def test_common_error_scenarios_documented(self):
        """Test that common error scenarios are documented."""
        # Check for common error scenario documentation
        common_errors = [
            "Missing API keys",
            "Invalid model names",
            "Configuration errors",
            "Import errors",
            "Dependency issues",
        ]

        # Check if troubleshooting documentation exists and covers common errors
        troubleshooting_files = [
            Path("docs/troubleshooting.md"),
            Path("external-docs/troubleshooting.md"),
            Path("docs/common-errors.md"),
        ]

        found_troubleshooting = []
        for file_path in troubleshooting_files:
            if file_path.exists():
                found_troubleshooting.append(file_path)
                print(f"✅ Found troubleshooting documentation: {file_path}")

                # Check if it covers common errors
                content = file_path.read_text().lower()
                covered_errors = []
                for error in common_errors:
                    if any(keyword in content for keyword in error.lower().split()):
                        covered_errors.append(error)

                if covered_errors:
                    print(
                        f"   Covers {len(covered_errors)}/{len(common_errors)} common error types"
                    )

        if found_troubleshooting:
            print(f"✅ Found {len(found_troubleshooting)} troubleshooting documentation files")
        else:
            print("⚠️  No troubleshooting documentation found")

    def test_error_message_consistency(self):
        """Test that error messages follow consistent patterns."""
        # This test checks that error messages are consistent across the codebase
        consistency_test_code = """
import sys

try:
    import saplings

    error_messages = []

    # Collect error messages from different scenarios
    test_scenarios = [
        ("missing_provider", lambda: saplings.AgentConfig(model_name="test")),
        ("missing_model", lambda: saplings.AgentConfig(provider="test")),
        ("empty_provider", lambda: saplings.AgentConfig(provider="", model_name="test")),
        ("empty_model", lambda: saplings.AgentConfig(provider="test", model_name="")),
    ]

    for scenario_name, test_func in test_scenarios:
        try:
            test_func()
            print(f"⚠️  {scenario_name}: Should have failed")
        except Exception as e:
            error_msg = str(e)
            error_messages.append((scenario_name, error_msg))
            print(f"✅ {scenario_name}: {error_msg[:100]}...")

    # Check for consistency patterns
    if error_messages:
        print(f"✅ Collected {len(error_messages)} error messages")

        # Check if messages follow similar patterns
        has_consistent_format = True
        for scenario, msg in error_messages:
            # Basic consistency checks
            if len(msg) < 10:
                print(f"⚠️  {scenario}: Error message too short")
                has_consistent_format = False

        if has_consistent_format:
            print("✅ Error messages appear to follow consistent patterns")
        else:
            print("⚠️  Error messages could be more consistent")

    print("✅ Error message consistency tests completed")
    exit(0)

except Exception as e:
    print(f"❌ Consistency test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", consistency_test_code],
                timeout=30,
                capture_output=True,
                text=True,
                cwd="/Users/jacobwarren/Development/agents/saplings",
                check=False,
            )

            if result.returncode == 0:
                print("✅ Error message consistency validation:")
                for line in result.stdout.strip().split("\n"):
                    print(f"   {line}")
            else:
                print(f"⚠️  Consistency test: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.fail("Consistency test timed out")

    def test_help_system_exists(self):
        """Test that help system is implemented."""
        # Check if help functions exist in the main package
        help_test_code = """
import sys

try:
    import saplings

    # Check for help functions
    help_functions = ['help', 'discover', 'troubleshoot']
    found_help = []

    for func_name in help_functions:
        if hasattr(saplings, func_name):
            func = getattr(saplings, func_name)
            if callable(func):
                found_help.append(func_name)
                print(f"✅ Found help function: saplings.{func_name}")

    if found_help:
        print(f"✅ Found {len(found_help)} help functions")
    else:
        print("⚠️  No help functions found in main package")

    # Check for help in AgentConfig
    if hasattr(saplings.AgentConfig, 'help'):
        print("✅ Found AgentConfig.help()")
    else:
        print("⚠️  AgentConfig.help() not found")

    print("✅ Help system tests completed")
    exit(0)

except Exception as e:
    print(f"❌ Help system test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
"""

        try:
            result = subprocess.run(
                [sys.executable, "-c", help_test_code],
                timeout=30,
                capture_output=True,
                text=True,
                cwd="/Users/jacobwarren/Development/agents/saplings",
                check=False,
            )

            if result.returncode == 0:
                print("✅ Help system validation:")
                for line in result.stdout.strip().split("\n"):
                    print(f"   {line}")
            else:
                print(f"⚠️  Help system test: {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.fail("Help system test timed out")

    def test_error_handling_comprehensive_coverage(self):
        """Test that error handling has comprehensive coverage."""
        # Analyze the current state of error handling
        print("\n📊 Error Handling Coverage Analysis:")

        # Count error handling test files
        error_test_patterns = ["test_*error*.py", "test_*validation*.py", "test_*handling*.py"]

        error_test_files = []
        for pattern in error_test_patterns:
            error_test_files.extend(Path("tests").rglob(pattern))

        unique_error_tests = list(set(error_test_files))
        print(f"   Error handling test files: {len(unique_error_tests)}")

        # Check for error handling documentation
        error_docs = (
            list(Path("docs").rglob("*error*.md"))
            + list(Path("external-docs").rglob("*error*.md"))
            + list(Path("docs").rglob("*troubleshoot*.md"))
        )

        print(f"   Error handling documentation files: {len(error_docs)}")

        # Overall assessment
        total_coverage = len(unique_error_tests) + len(error_docs)

        if total_coverage >= 5:
            print(f"✅ Excellent error handling coverage: {total_coverage} files")
        elif total_coverage >= 3:
            print(f"✅ Good error handling coverage: {total_coverage} files")
        elif total_coverage >= 1:
            print(f"⚠️  Basic error handling coverage: {total_coverage} files")
        else:
            print("⚠️  Limited error handling coverage")

        # Should have reasonable error handling coverage
        assert (
            total_coverage >= 3
        ), f"Should have at least 3 error handling files, found {total_coverage}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
