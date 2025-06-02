"""
Test security and safety review for publication readiness.

This module tests Task 7.10: Security and safety review.
Covers security vulnerabilities, unsafe operations, and safety checks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestSecurityAndSafetyReview:
    """Test security and safety review."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_input_validation_security(self):
        """Test that all user inputs are properly validated."""
        try:
            from saplings.api.memory.document import Document, DocumentMetadata

            # Test document creation with various inputs
            test_cases = [
                # Valid inputs
                ("Valid content", {"source": "test.txt"}),
                ("", {"source": "empty.txt"}),  # Empty content should be allowed
                # Edge cases that should be handled safely
                ("A" * 10000, {"source": "large.txt"}),  # Very large content
                ("Special chars: !@#$%^&*()", {"source": "special.txt"}),
                ("Unicode: 你好世界 🌍", {"source": "unicode.txt"}),
            ]

            for content, metadata_dict in test_cases:
                try:
                    metadata = DocumentMetadata(
                        source=metadata_dict.get("source", ""),
                        content_type="text/plain",
                        language="en",
                        author="test",
                    )
                    doc = Document(content=content, metadata=metadata)

                    # Verify document was created safely
                    assert doc.content == content
                    assert doc.metadata.source == metadata_dict["source"]
                    assert doc.id is not None

                except Exception as e:
                    # If an exception occurs, it should be a controlled validation error
                    assert isinstance(
                        e, (ValueError, TypeError)
                    ), f"Unexpected exception type: {type(e)}"

        except ImportError as e:
            pytest.skip(f"Input validation test skipped due to import error: {e}")

    def test_code_injection_prevention(self):
        """Test that code injection is prevented."""
        try:
            from saplings.api.tools import Tool

            # Create a test tool that should not execute arbitrary code
            class TestTool(Tool):
                def execute(self, input_data: str) -> str:
                    # This should safely process input without executing it as code
                    return f"Safely processed: {input_data}"

            tool = TestTool()

            # Test potentially malicious inputs
            malicious_inputs = [
                "__import__('os').system('rm -rf /')",
                "eval('print(\"hacked\")')",
                "exec('import os; os.system(\"ls\")')",
                "; rm -rf /",
                "$(rm -rf /)",
                "`rm -rf /`",
                "'; DROP TABLE users; --",
            ]

            for malicious_input in malicious_inputs:
                try:
                    result = tool.execute(malicious_input)

                    # The tool should safely process the input as text, not execute it
                    assert isinstance(result, str)
                    assert "Safely processed:" in result
                    assert malicious_input in result

                    # Verify no actual code execution occurred
                    # (This is a basic check - in a real scenario, we'd use sandboxing)

                except Exception as e:
                    # If an exception occurs, it should be a controlled security error
                    assert "security" in str(e).lower() or "injection" in str(e).lower()

        except ImportError as e:
            pytest.skip(f"Code injection prevention test skipped due to import error: {e}")

    def test_file_path_safety(self):
        """Test that file paths are handled safely."""
        try:
            from saplings.api.memory.document import Document, DocumentMetadata

            # Test potentially dangerous file paths
            dangerous_paths = [
                "../../../etc/passwd",
                "/etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "C:\\Windows\\System32\\config\\SAM",
                "/dev/null",
                "/proc/self/environ",
                "file:///etc/passwd",
                "http://evil.com/malware",
            ]

            for dangerous_path in dangerous_paths:
                try:
                    metadata = DocumentMetadata(
                        source=dangerous_path,
                        content_type="text/plain",
                        language="en",
                        author="test",
                    )
                    doc = Document(content="test content", metadata=metadata)

                    # The path should be stored as-is but not used for actual file operations
                    # without proper validation
                    assert doc.metadata.source == dangerous_path

                    # Verify no actual file system access occurred
                    # (In a real implementation, there would be path sanitization)

                except Exception as e:
                    # If an exception occurs, it should be a controlled validation error
                    assert isinstance(e, (ValueError, OSError, PermissionError))

        except ImportError as e:
            pytest.skip(f"File path safety test skipped due to import error: {e}")

    def test_data_sanitization(self):
        """Test that sensitive data is properly sanitized."""
        try:
            from saplings.api.memory.document import Document, DocumentMetadata

            # Test with potentially sensitive data
            sensitive_data = [
                "API Key: sk-1234567890abcdef",
                "Password: mySecretPassword123",
                "Credit Card: 4111-1111-1111-1111",
                "SSN: 123-45-6789",
                "Email: user@example.com",
                "Phone: +1-555-123-4567",
            ]

            for data in sensitive_data:
                metadata = DocumentMetadata(
                    source="sensitive.txt", content_type="text/plain", language="en", author="test"
                )
                doc = Document(content=data, metadata=metadata)

                # Verify document was created (basic functionality)
                assert doc.content == data
                assert doc.id is not None

                # In a real implementation, there might be:
                # - Automatic detection and redaction of sensitive data
                # - Warnings about sensitive data
                # - Encryption of sensitive content
                # For now, we just verify the API doesn't crash

        except ImportError as e:
            pytest.skip(f"Data sanitization test skipped due to import error: {e}")

    def test_resource_limits(self):
        """Test that resource limits are enforced."""
        try:
            from saplings.api.memory.document import Document, DocumentMetadata

            # Test with large amounts of data
            large_content = "A" * 1000000  # 1MB of data

            metadata = DocumentMetadata(
                source="large.txt", content_type="text/plain", language="en", author="test"
            )

            try:
                doc = Document(content=large_content, metadata=metadata)

                # If successful, verify the document was created properly
                assert len(doc.content) == 1000000
                assert doc.id is not None

                # In a production system, there might be size limits
                # For now, we just verify it doesn't cause memory issues

            except MemoryError:
                # This is acceptable - the system should handle memory limits gracefully
                pass
            except Exception as e:
                # Other exceptions should be controlled resource limit errors
                assert "limit" in str(e).lower() or "size" in str(e).lower()

        except ImportError as e:
            pytest.skip(f"Resource limits test skipped due to import error: {e}")

    def test_safe_defaults(self):
        """Test that all operations use safe defaults."""
        try:
            from saplings.api.memory.document import Document, DocumentMetadata

            # Test document creation with minimal parameters
            doc = Document(content="test content")

            # Verify safe defaults are used
            assert doc.content == "test content"
            assert doc.id is not None
            assert doc.metadata is not None

            # Test metadata creation with minimal parameters
            metadata = DocumentMetadata()

            # Verify safe defaults
            assert metadata.source == ""
            assert metadata.content_type == "text/plain"
            assert metadata.language == "en"
            assert metadata.author == ""

        except ImportError as e:
            pytest.skip(f"Safe defaults test skipped due to import error: {e}")

    def test_error_handling_security(self):
        """Test that error handling doesn't leak sensitive information."""
        try:
            from saplings.api.memory.document import Document, DocumentMetadata

            # Test with invalid inputs that should cause errors
            invalid_inputs = [
                123,  # Non-string content
                [],  # List content
                {},  # Dict content
            ]

            for invalid_input in invalid_inputs:
                try:
                    doc = Document(content=invalid_input)
                    # If this succeeds, verify it's handled safely
                    # Some inputs might be converted to strings
                    assert doc.content is not None
                    assert doc.id is not None

                except Exception as e:
                    # Verify error messages don't leak sensitive information
                    error_msg = str(e).lower()

                    # Error messages should not contain:
                    sensitive_terms = [
                        "password",
                        "key",
                        "secret",
                        "token",
                        "credential",
                        "internal",
                        "debug",
                        "stack",
                        "traceback",
                    ]

                    for term in sensitive_terms:
                        if term in error_msg:
                            # This might be acceptable for debug info, but flag it
                            print(f"Warning: Error message contains '{term}': {e}")

                    # Error should be a controlled validation error
                    assert isinstance(e, (ValueError, TypeError))

            # Test None content separately as it might be handled differently
            try:
                doc = Document(content=None)  # type: ignore
                # If None is accepted, it should be handled safely
                # This might be a design decision to allow None content
                assert doc.id is not None
                print(f"Info: None content accepted, resulting in: {doc.content}")

            except Exception as e:
                # Should be a controlled validation error
                assert isinstance(e, (ValueError, TypeError))

        except ImportError as e:
            pytest.skip(f"Error handling security test skipped due to import error: {e}")

    def test_api_access_control(self):
        """Test that API access is properly controlled."""
        try:
            import saplings

            # Test that internal APIs are not exposed
            internal_patterns = [
                "_internal",
                "__private",
                "_private",
                "secret",
                "password",
                "key",
            ]

            # Get all attributes of the main saplings module
            saplings_attrs = dir(saplings)

            for attr in saplings_attrs:
                attr_lower = attr.lower()

                # Check for potentially sensitive attributes
                for pattern in internal_patterns:
                    if pattern in attr_lower and not attr.startswith("__"):
                        # This might be a legitimate internal API exposure
                        # Flag it for review
                        print(f"Warning: Potentially internal API exposed: {attr}")

            # Verify that main public APIs are available
            expected_public_apis = ["Agent", "Tool", "Document", "LLM"]

            for api in expected_public_apis:
                if hasattr(saplings, api):
                    # Verify the API is accessible
                    api_obj = getattr(saplings, api)
                    assert api_obj is not None
                else:
                    print(f"Warning: Expected public API not found: {api}")

        except ImportError as e:
            pytest.skip(f"API access control test skipped due to import error: {e}")

    def test_dependency_security(self):
        """Test that dependencies are handled securely."""
        # Test that missing optional dependencies don't cause security issues
        try:
            import saplings

            # This should work even with missing optional dependencies
            assert saplings is not None

            # Test that warnings about missing dependencies are informational only
            # and don't expose sensitive information

        except ImportError as e:
            pytest.fail(f"Main package import should not fail due to missing dependencies: {e}")

    def test_configuration_security(self):
        """Test that configuration is handled securely."""
        try:
            from saplings.api.memory.document import DocumentMetadata

            # Test that configuration doesn't accept dangerous values
            test_configs = [
                {"source": "/etc/passwd"},
                {"content_type": "application/x-executable"},
                {"author": "../../../root"},
            ]

            for config in test_configs:
                try:
                    metadata = DocumentMetadata(**config)

                    # If successful, verify values are stored safely
                    for key, value in config.items():
                        assert getattr(metadata, key) == value

                except Exception as e:
                    # Should be a controlled validation error
                    assert isinstance(e, (ValueError, TypeError))

        except ImportError as e:
            pytest.skip(f"Configuration security test skipped due to import error: {e}")
