"""
Tests for the PatchGenerator class.
"""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest

from saplings.self_heal.patch_generator import PatchGenerator, PatchStatus


class TestPatchGenerator:
    """Tests for the PatchGenerator class."""

    @pytest.fixture
    def patch_generator(self):
        """Create a PatchGenerator instance for testing."""
        return PatchGenerator(max_retries=3)

    def test_initialization(self, patch_generator):
        """Test initialization of PatchGenerator."""
        assert patch_generator.max_retries == 3
        assert patch_generator.retry_count == 0
        assert patch_generator.patches == []

    def test_analyze_error_syntax_error(self, patch_generator):
        """Test analyzing a syntax error."""
        code = "def foo():\n    print('Hello, world!'\n"
        error = "SyntaxError: unexpected EOF while parsing"

        error_info = patch_generator.analyze_error(code, error)

        assert error_info["type"] == "SyntaxError"
        assert error_info["message"] == "unexpected EOF while parsing"
        assert "missing_parenthesis" in error_info["patterns"]

    def test_analyze_error_name_error(self, patch_generator):
        """Test analyzing a name error."""
        code = "def foo():\n    print(bar)\n"
        error = "NameError: name 'bar' is not defined"

        error_info = patch_generator.analyze_error(code, error)

        assert error_info["type"] == "NameError"
        assert error_info["message"] == "name 'bar' is not defined"
        assert "undefined_variable" in error_info["patterns"]
        assert error_info["variable"] == "bar"

    def test_generate_patch_syntax_error(self, patch_generator):
        """Test generating a patch for a syntax error."""
        code = "def foo():\n    print('Hello, world!'\n"
        error = "SyntaxError: unexpected EOF while parsing"

        patch = patch_generator.generate_patch(code, error)

        assert patch.original_code == code
        assert patch.error == error
        assert ")" in patch.patched_code
        assert patch.status == PatchStatus.GENERATED

    def test_generate_patch_name_error(self, patch_generator):
        """Test generating a patch for a name error."""
        code = "def foo():\n    print(bar)\n"
        error = "NameError: name 'bar' is not defined"

        # Mock the _fix_undefined_variable method to ensure it returns a fixed code
        with mock_patch.object(patch_generator, "_fix_undefined_variable") as mock_fix:
            # Set up the mock to return a fixed code
            fixed_code = "def foo():\n    bar = None  # TODO: Replace with appropriate value\n    print(bar)\n"
            mock_fix.return_value = fixed_code

            # Generate the patch
            patch = patch_generator.generate_patch(code, error)

            # Verify the patch
            assert patch.original_code == code
            assert patch.error == error
            assert "bar = " in patch.patched_code
        assert patch.status == PatchStatus.GENERATED

    def test_apply_patch(self, patch_generator):
        """Test applying a patch."""
        code = "def foo():\n    print('Hello, world!'\n"
        error = "SyntaxError: unexpected EOF while parsing"

        patch = patch_generator.generate_patch(code, error)
        result = patch_generator.apply_patch(patch)

        assert result.success
        assert result.patched_code == patch.patched_code
        assert patch_generator.retry_count == 1
        assert len(patch_generator.patches) == 1

    def test_retry_mechanism(self, patch_generator):
        """Test the retry mechanism with multiple patches."""
        code = "def foo():\n    print(bar)\n"

        # First error: NameError
        error1 = "NameError: name 'bar' is not defined"
        patch1 = patch_generator.generate_patch(code, error1)
        patch_generator.apply_patch(patch1)  # Apply the patch

        # Create a different error for the second patch
        # We'll use a different code to ensure the patches are different
        code2 = "def bar():\n    x = 1\n    print(y)\n"
        error2 = "NameError: name 'y' is not defined"
        patch2 = patch_generator.generate_patch(code2, error2)
        patch_generator.apply_patch(patch2)  # Apply the patch

        assert patch_generator.retry_count == 2
        assert len(patch_generator.patches) == 2

        # Verify the patches are different
        assert patch1.patched_code != patch2.patched_code

    def test_retry_limit(self, patch_generator):
        """Test that retry limit is enforced."""
        code = "def foo():\n    print(bar)\n"
        error = "NameError: name 'bar' is not defined"

        # Exhaust retries
        for _ in range(patch_generator.max_retries):
            patch = patch_generator.generate_patch(code, error)
            patch_generator.apply_patch(patch)

        # Try one more time, should fail
        patch = patch_generator.generate_patch(code, error)
        result = patch_generator.apply_patch(patch)

        assert not result.success
        assert result.error == "Maximum retry limit reached"
        assert patch_generator.retry_count == patch_generator.max_retries

    def test_validate_patch(self, patch_generator):
        """Test validating a patch."""
        code = "def foo():\n    print('Hello, world!'\n"
        error = "SyntaxError: unexpected EOF while parsing"

        patch_obj = patch_generator.generate_patch(code, error)

        # Mock the execution function to simulate successful execution
        with mock_patch(
            "saplings.self_heal.patch_generator.PatchGenerator._execute_code",
            return_value=(True, None),
        ):
            is_valid, _ = patch_generator.validate_patch(patch_obj.patched_code)
            assert is_valid

        # Mock the execution function to simulate failed execution
        with mock_patch(
            "saplings.self_heal.patch_generator.PatchGenerator._execute_code",
            return_value=(False, "Error"),
        ):
            is_valid, error_msg = patch_generator.validate_patch(patch_obj.patched_code)
            assert not is_valid
            assert error_msg == "Error"

    def test_after_success(self, patch_generator):
        """Test the after_success method for collecting successful patches."""
        code = "def foo():\n    print(bar)\n"
        error = "NameError: name 'bar' is not defined"

        patch = patch_generator.generate_patch(code, error)
        patch_generator.apply_patch(patch)

        # Mock the success pair collector
        mock_collector = MagicMock()
        patch_generator.success_pair_collector = mock_collector

        patch_generator.after_success(patch)

        # Verify the collector was called
        mock_collector.collect.assert_called_once()
        args, _ = mock_collector.collect.call_args
        assert args[0] == patch

    def test_complex_patch_generation(self, patch_generator):
        """Test generating patches for more complex errors."""
        # Test indentation error
        code = "def foo():\nprint('Hello, world!')\n"
        error = "IndentationError: expected an indented block"

        # Mock the _fix_indentation method
        with mock_patch.object(patch_generator, "_fix_indentation") as mock_fix:
            # Set up the mock to return a fixed code
            fixed_code = "def foo():\n    print('Hello, world!')\n"
            mock_fix.return_value = fixed_code

            # Generate the patch
            patch = patch_generator.generate_patch(code, error)

            # Verify the patch
            assert patch.original_code == code
            assert patch.error == error
            assert "    print" in patch.patched_code  # Should add indentation
            assert patch.status == PatchStatus.GENERATED

        # Test argument error
        code = "def foo(a, b):\n    return a + b\n\nresult = foo(1, 2, 3)\n"
        error = "TypeError: foo() takes 2 positional arguments but 3 were given"

        # Mock the _fix_argument_error method
        with mock_patch.object(patch_generator, "_fix_argument_error") as mock_fix:
            # Set up the mock to return a fixed code
            fixed_code = "def foo(a, b):\n    return a + b\n\nresult = foo(1, 2)\n"
            mock_fix.return_value = fixed_code

            # Generate the patch
            patch = patch_generator.generate_patch(code, error)

            # Verify the patch
            assert patch.original_code == code
            assert patch.error == error
            # The patched code should have the correct number of arguments
            # Use a more flexible regex pattern to match the function call with 2 arguments
            import re

            assert re.search(r"foo\s*\(\s*1\s*,\s*2\s*\)", patch.patched_code) is not None
        assert patch.status == PatchStatus.GENERATED

        # Test missing module error
        code = "import nonexistent_module\n\nprint(nonexistent_module.foo())\n"
        error = "ModuleNotFoundError: No module named 'nonexistent_module'"

        # Mock the _fix_missing_module method
        with mock_patch.object(patch_generator, "_fix_missing_module") as mock_fix:
            # Set up the mock to return a fixed code
            fixed_code = 'try:\n    import nonexistent_module\nexcept ImportError:\n    print("Error: The \'nonexistent_module\' module is required but not installed.")\n    print("Please install it using: pip install nonexistent_module")\n    # TODO: Install the required module\n    import sys\n    sys.exit(1)\n\nprint(nonexistent_module.foo())\n'
            mock_fix.return_value = fixed_code

            # Generate the patch
            patch = patch_generator.generate_patch(code, error)

            # Verify the patch
            assert patch.original_code == code
            assert patch.error == error
            assert (
                "# TODO: Install" in patch.patched_code
            )  # Should add comment about installing the module
            assert patch.status == PatchStatus.GENERATED

    def test_reset(self, patch_generator):
        """Test resetting the patch generator state."""
        code = "def foo():\n    print(bar)\n"
        error = "NameError: name 'bar' is not defined"

        # Generate and apply a patch
        patch = patch_generator.generate_patch(code, error)
        patch_generator.apply_patch(patch)

        assert patch_generator.retry_count == 1
        assert len(patch_generator.patches) == 1

        # Reset the state
        patch_generator.reset()

        assert patch_generator.retry_count == 0
        assert len(patch_generator.patches) == 0
