from __future__ import annotations

"""
Tests for the sanitizer module.
"""


from saplings.security.sanitizer import sanitize_document_collection, sanitize_prompt


def test_sanitize_prompt_urls() -> None:
    """Test that URLs are properly sanitized."""
    input_text = "Check this website: https://example.com/path?query=param#fragment"
    sanitized = sanitize_prompt(input_text)
    assert "https://" not in sanitized
    assert "[URL]" in sanitized


def test_sanitize_prompt_shell_metacharacters() -> None:
    """Test that shell metacharacters are removed."""
    input_text = "rm -rf / ; ls | grep data > file.txt"
    sanitized = sanitize_prompt(input_text)
    assert ";" not in sanitized
    assert "|" not in sanitized
    assert ">" not in sanitized
    assert "rm -rf" not in sanitized
    assert "[COMMAND]" in sanitized
    assert "ls" in sanitized
    assert "grep data" in sanitized
    assert "file.txt" in sanitized


def test_sanitize_prompt_sql_injection() -> None:
    """Test that SQL injection patterns are sanitized."""
    input_text = "username'; DROP TABLE users; --"
    sanitized = sanitize_prompt(input_text)
    assert "DROP TABLE" not in sanitized
    assert "[SQL]" in sanitized


def test_sanitize_prompt_path_traversal() -> None:
    """Test that path traversal patterns are removed."""
    input_text = "../../etc/passwd"
    sanitized = sanitize_prompt(input_text)
    assert "../" not in sanitized
    assert "etc/passwd" in sanitized


def test_sanitize_prompt_html() -> None:
    """Test that HTML tags are removed."""
    input_text = "<script>alert('XSS')</script><img src=x onerror=alert('XSS')>"
    sanitized = sanitize_prompt(input_text)
    assert "<script>" not in sanitized
    assert "</script>" not in sanitized
    assert "<img" not in sanitized
    # We don't need to check for the exact content, just that the tags are removed
    assert "script" in sanitized
    assert "alert" in sanitized


def test_sanitize_prompt_max_length() -> None:
    """Test that prompts are truncated at max_len."""
    input_text = "a" * 10000
    sanitized = sanitize_prompt(input_text, max_len=5000)
    assert len(sanitized) == 5000


def test_sanitize_prompt_custom_patterns() -> None:
    """Test that custom patterns are removed."""
    import re

    custom_patterns = [re.compile(r"password\s*=\s*['\"].*?['\"]")]
    input_text = "The config has password = 'secret123'"
    sanitized = sanitize_prompt(input_text, custom_patterns=custom_patterns)
    assert "password = 'secret123'" not in sanitized


def test_sanitize_document_collection() -> None:
    """Test sanitizing a collection of documents."""
    documents = [
        "Doc 1 with https://example.com",
        "Doc 2 with rm -rf / ; ls",
        "Doc 3 with SELECT * FROM users",
    ]
    sanitized = sanitize_document_collection(documents)
    assert len(sanitized) == 3
    assert "https://" not in sanitized[0]
    assert ";" not in sanitized[1]
    assert "SELECT * FROM users" not in sanitized[2]
